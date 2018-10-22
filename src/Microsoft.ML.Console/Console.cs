// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.PCA;
using System;
using System.IO;
using System.Threading;

namespace Microsoft.ML.Runtime.Tools.Console2
{
    public static class Console2
    {
        private const string _directory = @"e:/src\MLNet/";

        //public static int Main(string[] args) => Maml.Main(args);

        private static string EscapePath(string path)
        {
            return path.Replace("\\", "\\\\");
        }

        private static void TestEntryPointRoutine(IHostEnvironment env, int threadIndex,
            string dataFile, string trainerName, string loader = null, string trainerArgs = null)
        {
            //var dataPath = GetDataPath(dataFile);
            var dataPath = Path.Join(_directory, dataFile);
            var outputPath = Path.Join(_directory, $"model{threadIndex}.zip");

            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1'
                        {3}
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {{
                      'Name': '{2}',
                      'Inputs': {{
                        'TrainingData': '$data1'
                         {4}
                      }},
                      'Outputs': {{
                        'PredictorModel': '$model'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'model' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(outputPath), trainerName,
                string.IsNullOrWhiteSpace(loader) ? "" : string.Format(",'CustomSchema': '{0}'", loader),
                string.IsNullOrWhiteSpace(trainerArgs) ? "" : trainerArgs
                );

            var jsonPath = Path.Join(_directory, $"graph{threadIndex}.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(env, args);
            cmd.Run();
        }

        public static TEnvironment AddStandardComponents<TEnvironment>(this TEnvironment env)
    where TEnvironment : IHostEnvironment
        {
            env.ComponentCatalog.RegisterAssembly(typeof(TextLoader).Assembly); // ML.Data
            env.ComponentCatalog.RegisterAssembly(typeof(LinearPredictor).Assembly); // ML.StandardLearners
            env.ComponentCatalog.RegisterAssembly(typeof(CategoricalTransform).Assembly); // ML.Transforms
            env.ComponentCatalog.RegisterAssembly(typeof(FastTreeBinaryPredictor).Assembly); // ML.FastTree
            env.ComponentCatalog.RegisterAssembly(typeof(EnsemblePredictor).Assembly); // ML.Ensemble
            env.ComponentCatalog.RegisterAssembly(typeof(KMeansPredictor).Assembly); // ML.KMeansClustering
            env.ComponentCatalog.RegisterAssembly(typeof(PcaPredictor).Assembly); // ML.PCA
            env.ComponentCatalog.RegisterAssembly(typeof(Experiment).Assembly); // ML.Legacy
            return env;
        }

        public static void Main()
        {
            var ts = Utils.BuildArray(16, t => new Thread(
                () =>
                {
                    for (long i = 0; ; ++i)
                    {
                        var env = new ConsoleEnvironment(42).AddStandardComponents();
                        Console.WriteLine($"TRIAL {i} for {t}");
                        TestEntryPointRoutine(env, t, "test/data/iris.txt", "Trainers.StochasticDualCoordinateAscentClassifier");
                    }
                }
                ));

            foreach (var t in ts)
                t.Start();
        }
    }
}