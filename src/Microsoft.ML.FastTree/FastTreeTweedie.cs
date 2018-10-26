// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;
using System.Linq;
using System.Text;

[assembly: LoadableClass(FastTreeTweedieTrainer.Summary, typeof(FastTreeTweedieTrainer), typeof(FastTreeTweedieTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer), typeof(SignatureFeatureScorerTrainer) },
    FastTreeTweedieTrainer.UserNameValue,
    FastTreeTweedieTrainer.LoadNameValue,
    FastTreeTweedieTrainer.ShortName)]

[assembly: LoadableClass(typeof(FastTreeTweediePredictor), null, typeof(SignatureLoadModel),
    "FastTree Tweedie Regression Executor",
    FastTreeTweediePredictor.LoaderSignature)]

namespace Microsoft.ML.Trainers.FastTree
{
    // The Tweedie boosting model follows the mathematics established in:
    // Yang, Quan, and Zou. "Insurance Premium Prediction via Gradient Tree-Boosted Tweedie Compound Poisson Models."
    // https://arxiv.org/pdf/1508.06378.pdf
    /// <include file='doc.xml' path='doc/members/member[@name="FastTreeTweedieRegression"]/*' />
    public sealed partial class FastTreeTweedieTrainer
         : BoostingFastTreeTrainerBase<FastTreeTweedieTrainer.Arguments, RegressionPredictionTransformer<FastTreeTweediePredictor>, FastTreeTweediePredictor>
    {
        internal const string LoadNameValue = "FastTreeTweedieRegression";
        internal const string UserNameValue = "FastTree (Boosted Trees) Tweedie Regression";
        internal const string Summary = "Trains gradient boosted decision trees to fit target values using a Tweedie loss function. This learner is a generalization of Poisson, compound Poisson, and gamma regression.";
        internal const string ShortName = "fttweedie";

        private TestHistory _firstTestSetHistory;
        private Test _trainRegressionTest;
        private Test _testRegressionTest;

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        private SchemaShape.Column[] _outputColumns;

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeTweedieTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weightColumn">The name for the column containing the initial weight.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="minDocumentsInLeafs">The minimal number of documents allowed in a leaf of a regression tree, out of the subsampled data.</param>
        /// <param name="numLeaves">The max number of leaves in each regression tree.</param>
        /// <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public FastTreeTweedieTrainer(IHostEnvironment env,
            string labelColumn,
            string featureColumn,
            string weightColumn = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDocumentsInLeafs = Defaults.MinDocumentsInLeafs,
            double learningRate = Defaults.LearningRates,
            Action<Arguments> advancedSettings = null)
            : base(env, TrainerUtils.MakeR4ScalarLabel(labelColumn), featureColumn, weightColumn, null, numLeaves, numTrees, minDocumentsInLeafs, learningRate, advancedSettings)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));

            Initialize();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="FastTreeTweedieTrainer"/> by using the legacy <see cref="Arguments"/> class.
        /// </summary>
        internal FastTreeTweedieTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, TrainerUtils.MakeR4ScalarLabel(args.LabelColumn))
        {
            Initialize();
        }

        protected override FastTreeTweediePredictor TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var trainData = context.TrainingSet;

            ValidData = context.ValidationSet;

            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(trainData, nameof(trainData));
                trainData.CheckRegressionLabel();
                trainData.CheckFeatureFloatVector();
                trainData.CheckOptFloatWeight();
                FeatureCount = trainData.Schema.Feature.Type.ValueCount;
                ConvertData(trainData);
                TrainCore(ch);
            }
            return new FastTreeTweediePredictor(Host, state.TrainedEnsemble, state.FeatureCount, InnerArgs);
        }

        protected override void CheckArgs(IChannel ch)
        {
            Contracts.AssertValue(ch);

            base.CheckArgs(ch);

            // REVIEW: In order to properly support early stopping, the early stopping metric should be a subcomponent, not just
            // a simple integer, because the metric that we might want is parameterized by this floating point "index" parameter. For now
            // we just leave the existing regression checks, though with a warning.

            if (Args.EarlyStoppingMetrics > 0)
                ch.Warning("For Tweedie regression, early stopping does not yet use the Tweedie distribution.");

            ch.CheckUserArg((Args.EarlyStoppingRule == null && !Args.EnablePruning) || (Args.EarlyStoppingMetrics >= 1 && Args.EarlyStoppingMetrics <= 2), nameof(Args.EarlyStoppingMetrics),
                    "earlyStoppingMetrics should be 1 or 2. (1: L1, 2: L2)");
        }

        private protected override ObjectiveFunctionBase ConstructObjFunc(IChannel ch, TrainState state)
        {
            return new ObjectiveImpl(state.TrainSet, Args);
        }

        private protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch, TrainState state)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch, state);
            if (Args.UseLineSearch)
            {
                var lossCalculator = new RegressionTest(optimizationAlgorithm.TrainingScores);
                // REVIEW: We should make loss indices an enum in BinaryClassificationTest.
                // REVIEW: Nope, subcomponent.
                optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, 1 /*L2 error*/, Args.NumPostBracketSteps, Args.MinStepSize);
            }

            return optimizationAlgorithm;
        }

        /// <summary>
        /// Gets the regression labels that were stored in the dataset skeleton, or
        /// constructs them from the ratings if absent. This returns null if the
        /// dataset itself is null.
        /// </summary>
        /// <param name="set">The dataset</param>
        /// <returns>The list of regression targets, or null if <paramref name="set"/> was null</returns>
        public static float[] GetDatasetRegressionLabels(Dataset set)
        {
            if (set == null)
                return null;
            double[] dlabels = set.Targets;
            Contracts.AssertValue(dlabels);
            Contracts.Assert(dlabels.Length == set.NumDocs);
            // REVIEW: Seems wasteful??
            return Utils.BuildArray(dlabels.Length, i => (float)dlabels[i]);
        }

        private protected override void PrepareLabels(IChannel ch, TrainState state)
        {
        }

        private protected override Test ConstructTestForTrainingData(TrainState state)
        {
            return new RegressionTest(state.ConstructScoreTracker(state.TrainSet));
        }

        private void Initialize()
        {
            Host.CheckUserArg(1 <= Args.Index && Args.Index <= 2, nameof(Args.Index), "Must be in the range [1, 2]");

            _outputColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false)
            };
        }

        private void AddFullRegressionTests(TrainState state)
        {
            // Always compute training L1/L2 errors.
            state.Tests.Add(new RegressionTest(state.ConstructScoreTracker(state.TrainSet)));
            RegressionTest validTest = null;
            if (state.ValidSet != null)
            {
                validTest = new RegressionTest(state.ConstructScoreTracker(state.ValidSet));
                state.Tests.Add(validTest);
            }

            // If external label is missing use Rating column for L1/L2 error.
            // The values may not make much sense if regression value is not an actual label value.
            if (state.TestSets != null)
            {
                for (int t = 0; t < state.TestSets.Length; ++t)
                    state.Tests.Add(new RegressionTest(state.ConstructScoreTracker(state.TestSets[t])));
            }
        }

        private protected override void InitializeTests(TrainState state)
        {
            // Initialize regression tests.
            if (Args.TestFrequency != int.MaxValue)
                AddFullRegressionTests(state);

            if (Args.PrintTestGraph)
            {
                // If FirstTestHistory is null (which means the tests were not intialized due to /tf==infinity),
                // we need initialize first set for graph printing.
                // Adding to a tests would result in printing the results after final iteration.
                if (_firstTestSetHistory == null)
                {
                    var firstTestSetTest = new RegressionTest(state.ConstructScoreTracker(state.TestSets[0]));
                    _firstTestSetHistory = new TestHistory(firstTestSetTest, 0);
                }
            }

            if (Args.PrintTrainValidGraph && _trainRegressionTest == null)
            {
                Test trainRegressionTest = new RegressionTest(state.ConstructScoreTracker(state.TrainSet));
                _trainRegressionTest = trainRegressionTest;
            }

            if (Args.PrintTrainValidGraph && _testRegressionTest == null && Utils.Size(state.TestSets) > 0)
                _testRegressionTest = new RegressionTest(state.ConstructScoreTracker(state.TestSets[0]));

            // Add early stopping if appropriate.
            state.TrainTest = new RegressionTest(state.ConstructScoreTracker(state.TrainSet), Args.EarlyStoppingMetrics);
            if (state.ValidSet != null)
                state.ValidTest = new RegressionTest(state.ConstructScoreTracker(state.ValidSet), Args.EarlyStoppingMetrics);

            if (Args.EnablePruning && state.ValidTest != null)
            {
                if (Args.UseTolerantPruning) // Use simple early stopping condition.
                    state.PruningTest = new TestWindowWithTolerance(state.ValidTest, 0, Args.PruningWindowSize, Args.PruningThreshold);
                else
                    state.PruningTest = new TestHistory(state.ValidTest, 0);
            }
        }

        private protected override void PrintIterationMessage(IChannel ch, IProgressChannel pch, TrainState state)
        {
            base.PrintIterationMessage(ch, pch, state);
        }

        protected override string GetTestGraphHeader()
        {
            StringBuilder headerBuilder = new StringBuilder("Eval:\tFileName\tNDCG@1\tNDCG@2\tNDCG@3\tNDCG@4\tNDCG@5\tNDCG@6\tNDCG@7\tNDCG@8\tNDCG@9\tNDCG@10");

            if (Args.PrintTrainValidGraph)
            {
                headerBuilder.Append("\tNDCG@20\tNDCG@40");
                headerBuilder.Append("\nNote: Printing train L2 error as NDCG@20 and test L2 error as NDCG@40..\n");
            }

            return headerBuilder.ToString();
        }

        private protected override void ComputeTests(TrainState state)
        {
            if (_firstTestSetHistory != null)
                _firstTestSetHistory.ComputeTests();

            if (_trainRegressionTest != null)
                _trainRegressionTest.ComputeTests();

            if (_testRegressionTest != null)
                _testRegressionTest.ComputeTests();

            if (state.PruningTest != null)
                state.PruningTest.ComputeTests();
        }

        protected override string GetTestGraphLine()
        {
            StringBuilder lineBuilder = new StringBuilder();

            lineBuilder.AppendFormat("Eval:\tnet.{0:D8}.ini", Ensemble.NumTrees - 1);

            foreach (var r in _firstTestSetHistory.ComputeTests())
            {
                lineBuilder.AppendFormat("\t{0:0.0000}", r.FinalValue);
            }

            double trainRegression = 0.0;
            double validRegression = 0.0;

            // We only print non-zero train&valid graph if earlyStoppingTruncation!=0.
            // In case /es is not set, we print 0 for train and valid graph NDCG.
            // Let's keeping this behaviour for backward compatibility with previous FR version.
            // Ideally /graphtv should enforce non-zero /es in the commandline validation.
            if (_trainRegressionTest != null)
                trainRegression = _trainRegressionTest.ComputeTests().Last().FinalValue;
            if (_testRegressionTest != null)
                validRegression = _testRegressionTest.ComputeTests().Last().FinalValue;

            lineBuilder.AppendFormat("\t{0:0.0000}\t{1:0.0000}", trainRegression, validRegression);

            return lineBuilder.ToString();
        }

        protected override void Train(IChannel ch)
        {
            base.Train(ch);
            // Print final last iteration.
            // Note that trainNDCG printed in graph will be from copy of a value from previous iteration
            // and will differ slightly from the proper final value computed by FullTest.
            // We cannot compute the final NDCG here due to the fact we use FastNDCGTestForTrainSet
            // computing NDCG based on label sort saved during gradient computation (and we don't have
            // gradients for n+1 iteration).
            // Keeping it in sync with original FR code
            PrintTestGraph(ch);
        }

        protected override RegressionPredictionTransformer<FastTreeTweediePredictor> MakeTransformer(FastTreeTweediePredictor model, Schema trainSchema)
         => new RegressionPredictionTransformer<FastTreeTweediePredictor>(Host, model, trainSchema, FeatureColumn.Name);

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        private sealed class ObjectiveImpl : ObjectiveFunctionBase, IStepSearch
        {
            private readonly float[] _labels;
            private readonly Double _index1; // 1 minus the index parameter.
            private readonly Double _index2; // 2 minus the index parameter.
            private readonly Double _maxClamp;

            public ObjectiveImpl(Dataset trainData, Arguments args)
                : base(
                    trainData,
                    args.LearningRates,
                    args.Shrinkage,
                    args.MaxTreeOutput,
                    args.GetDerivativesSampleRate,
                    args.BestStepRankingRegressionTrees,
                    args.RngSeed)
            {
                if (args.DropoutRate > 0 && LearningRate > 0) // Don't do shrinkage if dropouts are used.
                    Shrinkage = 1.0 / LearningRate;

                _labels = GetDatasetRegressionLabels(trainData);
                // Should we fail instead of clamping negative values to 0?
                for (int i = 0; i < _labels.Length; ++i)
                {
                    if (_labels[i] < 0)
                        _labels[i] = 0;
                }

                _index1 = 1 - args.Index;
                _index2 = 2 - args.Index;
                _maxClamp = Math.Abs(args.MaxTreeOutput);
            }

            public void AdjustTreeOutputs(IChannel ch, RegressionTree tree, DocumentPartitioning partitioning, ScoreTracker trainingScores)
            {
                double shrinkage = LearningRate * Shrinkage;
                var scores = trainingScores.Scores;
                var weights = trainingScores.Dataset.SampleWeights;

                // Following equation 18, and line 2c of algorithm 1 in the source paper.
                for (int l = 0; l < tree.NumLeaves; ++l)
                {
                    Double num = 0;
                    Double denom = 0;

                    if (_index1 == 0)
                    {
                        // The index == 1 Poisson case.
                        foreach (int i in partitioning.DocumentsInLeaf(l))
                        {
                            var s = scores[i];
                            var w = weights == null ? 1 : weights[i];
                            num += w * _labels[i];
                            denom += w * Math.Exp(s);
                        }
                    }
                    else
                    {
                        // The index in (1,2] case.
                        foreach (int i in partitioning.DocumentsInLeaf(l))
                        {
                            var s = scores[i];
                            var w = weights == null ? 1 : weights[i];
                            num += w * _labels[i] * Math.Exp(_index1 * s);
                            denom += w * Math.Exp(_index2 * s);
                        }
                    }

                    var step = shrinkage * (Math.Log(num) - Math.Log(denom));
                    if (num == 0 && denom == 0)
                        step = 0;
                    // If we do not clamp, it is entirely possible for num to be 0 (with 0 labels), which
                    // means that we will have negative infinities in the leaf nodes. This has a number of
                    // bad negative effects we'd prefer to avoid. Nonetheless, we do give up a substantial
                    // amount of "gain" for those examples.
                    if (step < -_maxClamp)
                        step = -_maxClamp;
                    else if (step > _maxClamp)
                        step = _maxClamp;
                    tree.SetOutput(l, step);
                }
            }

            protected override void GetGradientInOneQuery(int query, int threadIndex)
            {
                int begin = Dataset.Boundaries[query];
                int end = Dataset.Boundaries[query + 1];

                // Gradient.
                unchecked
                {
                    if (_index1 == 0)
                    {
                        // In the case where index=1, the Tweedie distribution is Poisson. We treat this important
                        // class of distribution as a special case even though the mathematics wind up being
                        // the same.
                        for (int i = begin; i < end; ++i)
                        {
                            // From "Generalized Boosted Models: A guide to the gbm package" by Greg Ridgeway, 2007, section 4.7.
                            Gradient[i] = Math.Exp(Scores[i]) - _labels[i];
                        }
                    }
                    else
                    {
                        for (int i = begin; i < end; ++i)
                        {
                            // Following equation 14, and line 2a of algorithm 1 in the source paper. The w_i
                            // are not incorporated here.
                            Gradient[i] = Math.Exp(_index2 * Scores[i]) - _labels[i] * Math.Exp(_index1 * Scores[i]);
                        }
                    }
                }
            }
        }
    }

    public sealed class FastTreeTweediePredictor : FastTreePredictionWrapper
    {
        public const string LoaderSignature = "FastTreeTweedieExec";
        public const string RegistrationName = "FastTreeTweediePredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FTREE TW",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Add _defaultValueForMissing
                verWrittenCur: 0x00010003, // Categorical splits.
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FastTreeTweediePredictor).Assembly.FullName);
        }

        protected override uint VerNumFeaturesSerialized => 0x00010001;

        protected override uint VerDefaultValueSerialized => 0x00010002;

        protected override uint VerCategoricalSplitSerialized => 0x00010003;

        internal FastTreeTweediePredictor(IHostEnvironment env, Ensemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private FastTreeTweediePredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public static FastTreeTweediePredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new FastTreeTweediePredictor(env, ctx);
        }

        protected override void Map(ref VBuffer<float> src, ref float dst)
        {
            // The value learnt and predicted by the trees is the log of the expected value,
            // as seen in equation 9 of the paper. So for the actual prediction, we take its
            // exponent.
            base.Map(ref src, ref dst);
            // REVIEW: Some packages like R's GBM apparently clamp the input to the exponent
            // in the range [-19, 19]. We have historically taken a dim view of this sort of thing
            // ourselves, but if our views prove problematic we can reconsider. (An upper clamp of 19
            // seems far too restrictive, leading to a practical upper limit of about 178 million.)
            dst = MathUtils.ExpSlow(dst);
        }

        public override PredictionKind PredictionKind => PredictionKind.Regression;
    }

    public static partial class FastTree
    {
        [TlcModule.EntryPoint(Name = "Trainers.FastTreeTweedieRegressor",
            Desc = FastTreeTweedieTrainer.Summary,
            UserName = FastTreeTweedieTrainer.UserNameValue,
            ShortName = FastTreeTweedieTrainer.ShortName,
            XmlInclude = new [] { @"<include file='../Microsoft.ML.FastTree/doc.xml' path='doc/members/member[@name=""FastTreeTweedieRegression""]/*' />" })]
        public static CommonOutputs.RegressionOutput TrainTweedieRegression(IHostEnvironment env, FastTreeTweedieTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainTweeedie");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<FastTreeTweedieTrainer.Arguments, CommonOutputs.RegressionOutput>(host, input,
                () => new FastTreeTweedieTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.GroupIdColumn));
        }
    }
}
