# How To Read

In this document I review types in the `Microsoft.ML.Core` project and assembly, sharing thoughts on their ultimate fate (mostly whether they should be public or internal, but also thoughts on what we might want to do with them prior to ML.NET 1.0). This document is structured in a way that it is worthwhile first to understand how to read it. There is a tremendous amount of information to convey, and where possible I use jargon and shorthand to encode thoughts.

I only review currently public types. All types (even enums/delegates) are included.

Note that the headers *are* structured deliberately.

## Category Codes for Groups of Types

Quite naturally, the types of the assembly are best understood not individually, but by certain categories, since they often work together. For a given identified group, note that I will always provide some  "category codes." This is so one can understand at a glance how one should think about this set of types. These are the categories:

* **P** means public, that is, exposed to the user. Obviously these are the things requiring the most urgent review.
* **C** means component, that is, public-ish but is only useful to component authors. This indicates types or capabilities of no practical use to a user not writing their own components.
* **D** means something related specifically to dependency injection, or command line/entry-points specifically. Anything in **D** is implicitly in **C**, since the command-line/entry-point experience is one that we ought to consider out of scope for ML.NET 1.0. Nonetheless, is *is* important to retain, since we do not want to loose capability.
* **X** means something that ought to be removed. Can be repeated for emphasis.
* **M** means something that is worthwhile but still ought to be moved to a different assembly.

More rare tags:

* **I** means something that we may consider being purely internal, that is, not even available to component authors. There are few of these, since if something *could* be private or internal, we generally have already done so.
* **N** means something that the .NET team *might* want to consider elevating into the BCL. Interpret as "hey people, just casually consider this, no hard feelings if you don't" rather than any sort of "hard" suggestion.
* **U** means something "unusual" or "unnatural," that is, a situation where we, due perhaps to the peculiarities of our task, used the .NET framework in a way we're fairly certain it was not meant to be used in ways that terrify people that learn about it -- yet we cannot imagine a better way to achieve the same goals.
* **A** indicates something "architecturally tricky" that, while potentially serving a useful purpose, is I feel something ripe for redesign, yet perhaps don't have clear ideas yet on how to do so. Its absence doesn't mean that I consider something perfect. Rather, interpret it as something I am *especially* displeased with in its current form.

Sometimes an individual type in a group might have a different category, in which case this will be made clear in the individual type description by prefixing it with a parenthetical code. Not all types will have an individualized description, if they're sufficiently uninteresting or part of a group so insularly defined that it's not necessary to do so.

To give an example: a tag **PX** indicates something "public" that should be "removed." The suggestion there is, this is a type that due to the current way the system is architected simply cannot help but be exposed to the user, but I feel very strongly that the idiom is flawed in some way that we should strongly consider not allowing it to persist once ML.NET 1.0 is live, since the implication there is we'd live with that debt *forever*. (At least, forever in the context of a library.)

My expectation is that for ML.NET v1.0, **C** and **I** will functionally the same since extensibility of ML.NET is not a target scenario for v1.0. Nonetheless, since extensibility must be an eventual goal (if the framework is successful then it's only a matter of time before people ask, "hey, how do I write my own transformation of data"), it is still helpful to bear in mind.

## Interface Specific Notes

There is a desire among .NET people to limit the use of interfaces. For this reason, following every interface not marked **X** or **I** or **A** in the categories above, I give a number from 1 to 5 in parentheses, in the *suffix*. So: "`IFoo` (5)" should be interpreted as "I am quite certain `IFoo` can be turned into an abstract base class," while a (1) should be interpreted as "I am quite certain that this must remain an interface." Intermediate values represent uncertainty one way or the other. E.g., (3) would be, "might be nice, but I wouldn't be surprised if there are some reasons we cannot turn it into a base class."

For lower values where I suggest it remain and interface, there is another code immediately following the number to indicate why.

* **C** indicates that the interface is beneficial for covariance/contravariance capabilities. Pipeline/configuration components are common users of this interface tag.
* **M** indicates that the interface is beneficial from a multiple inheritance perspective.
* **R** indicates that the interface is beneficial because it allows us to restrict how an instance is used, in a way that subclassing could not reasonably do. (E.g., the `IReadOnlyDictionary<,>` application.) Strong overlap with **M**.
* **E** indicates that the interface is beneficial since usage patterns of implementing objects and objects that utilize the interface, make explicit interface declarations possible, which is critical to its intended usage in some way.
* **S** indicates that the interface was introduced because by making something an interface, any details of its actual implementation can appear in some separate code that does not pollute the code of the assembly.

So for example, I might say "`IBar` (2C)" to indicate, "I am fairly but not 100% sure this should remain an interface, since I think covariance is useful for this scenario."

# Namespace `Microsoft.ML.Runtime`

More or less the highest level namespace. Lots of important stuff here, but also lots of stuff that was only *previously* important. Also stuff people pushed here because they didn't know any better.

## Legacy Predictor

Category PX. Once upon a time there were `ITrainer`s that when trained produced `IPredictor`s, and these `IPredictor`s would be used to make predictions. This has not really been the case, yet we still use these interfaces. It may be that we will need *some* type to represent the artifact of training a model apart from the actual `ITransformer` instance, but what that is, is not yet clear to me.

* `IBulkDistributionPredictor<in TFeatures, in TFeaturesCollection, TResult, TResultCollection, out TResultDistribution, out TResultDistributionCollection>`
* `IBulkPredictor<in TFeatures, in TFeaturesCollection, out TResult, out TResultCollection>`
* `IDistPredictorProducing<out TResult, out TResultDistribution>`
* `IDistributionPredictor<in TFeatures, TResult, out TResultDistribution>`
* `IPredictor`
* `IPredictor<in TFeatures, out TResult>`
* `IPredictorProducing<out TResult>`

## Ensembles

Category C.

* `IModelCombiner<TModel, TPredictor>`, used by both FastTree and ensembles. I'm pretty sure this can be moved someplace less central.

## Host Environment

Category C. This requires some explanation, since previously this was P. Note that under issue [#1098](https://github.com/dotnet/machinelearning/issues/1098), an `IHostEnvironment` implementing object, the hypothetical `MLContext`, will be the user facing aspect to what we currently call `IHostEnvironment`. However, the fact that it *is* an `IHostEnvironment` will be a detail largely invisible to users. For the "public facing" aspect, see what is currently called the `HostEnvironmentBase` or `LocalEnvironment`, which will be morphed into `MLContext` in the coming weeks.

* `IHostEnvironment` (1EM), used to create `IHost`. Given the plan of the aforementioned issue, part of the "dual nature" of the object is to have an explicit interface implementation to hide the stuff from users, which means an interface is helpful and required here.
* `IHost` (4), spawned from `IHostEnvironment`, typically one of these is owned by a component (e.g., an `IDataView`).
* `IChannel` (4), spawned from `IHost`, typically one of these is owned by a *disposable* component (e.g., the `IRowCursor` out of an `IDataView`).
* `ChannelMessageKind`, an `enum` that's essentially a "log-level." Simple info messages have one kind, warnings have another, for example.
* `IChannelProvider` (1M), something that can yield `IChannel` instances.
* `IExceptionContext` (1M), all `IHostEnvironment`/`IHost`/`IChannel` descend from this. An interface that describes to `Contracts` some additional info to attach to messages.
* `IPipe<TMessage>` (3)
* `MessageSensitivity`, an `enum` that can be attached to a message to describe whether an error that is reported is safe or not. Currently underutilized. Requested as a high-priority ask by Office team, I did it right away, and the Office team promptly did absolutely nothing with this added capability.

Category CA. I feel like progress channels were fundamentally misdesigned. They ought to be designed more along the lines of server channels, which somehow have to communicate far, far more complex information but did so in a way that was much simpler. I don't think anyone quite understands them.

* `IProgressChannel`
* `IProgressChannelProvider`
* `IProgressEntry`
* `ProgressHeader`

Category C. Somewhat related and in the same category, but worth treating separately.

* `Contracts`, related to host environments via `IExceptionContext`, used for asserts and checks.
* `HostExtensions` used so that we don't have to specify a `MessageSensitivity` on every single message. Mostly extensions on `IChannel` despite the name.

### Server Channels

Category CU. Related to so-called TLC-TV scenarios, where there is higher levels of introspection possible on models as they're being trained.

* `ServerChannel`
* `ServerChannel.Bundle`
* `ServerChannel.IPendingBundleNotification` (4S)
* `ServerChannel.IServer` (4S)
* `ServerChannel.IServerFactory` (2XS)
* `ServerChannelUtilities`

### Telemetry

Category D. Produced only by commands, consumed only by the internal command line tool. The following are things reported by the commands (see `ICommand`) and are otherwise not used, since of course no telemetry is directly gathered for API users. Passed up the chain via `IPipe<>`, incidentally.

* `TelemetryException`
* `TelemetryMessage`
* `TelemetryMetric`
* `TelemetryTrace`

### Catalogs for Getting Dependency Injection Thingies

Category D.

* `ComponentCatalog`
* `ComponentCatalog.ComponentInfo`
* `ComponentCatalog.EntryPointInfo`
* `ComponentCatalog.LoadableClassInfo`

Category DX. I still think we can replace this sort of thing with MEF.

* `LoadableClassAttribute`
* `LoadableClassAttributeBase`

## Configuration

Category P. These so-called "component-factories" are an essential part of many objects' configurations. ML.NET is composable, but also because it is long running it is helpful to have an object that can "promise" to provide some implementation of an object, once certain key information becomes clear.  for example, one example I was looking at just today was [the `RffTransform`](https://github.com/dotnet/machinelearning/blob/fcea146add5e8b9c3197f800b7201b333ab5ae3e/src/Microsoft.ML.Transforms/RffTransform.cs#L44-L45), where the user can indicate what type of sampler they want to generate, but the transform only knows certain key parameters only after it has looked at the data.

* `IComponentFactory` [DX?]
* `IComponentFactory<out TComponent>` (1C)
* `IComponentFactory<in TArg1, out TComponent>` (1C)
* `IComponentFactory<in TArg1, in TArg2, out TComponent>` (1C)
* `IComponentFactory<in TArg1, in TArg2, in TArg3, out TComponent>` (1C)
* `ComponentFactoryUtils`, for easy creation of these objects.

## Sweeping

Category CXXX. The pipeline inference and sweeper assemblies should *not* be a part of ML.NET, at least, not yet until the project matures. Let those things live in some other repository. As near as I can tell, these types are here because the person working in pipeline inference and sweeping said, "hey, I have these types I want in both, let me just stuff them in this assembly that is referenced by both," and no responsible adult was around to tell them no. These types *absolutely do not belong in Core*.

* `IRunResult`
* `IRunResult<T>`
* `ISweeper`
* `ISweepResultEvaluator<in TResults>`
* `IParameterValue`
* `IParameterValue<out TValue>`
* `IValueGenerator`
* `ParameterSet`
* `RunMetric`
* `RunResult`

## File-ish Stream-ish

Category PX?N, *might* be able to transition P to D.

Used in the creation of components that take files.

We ought to investigate whether we can replace with `System.IO.FileInfo` or something. I guess *technically* this is more general since it's a more general way to open factories than just a `FileInfo`, but it's not clear to me how valuable this is. Someone on the efforts dealing with entry-points may be better suited to answer this question, since perhaps in those situations other less trivial implementations of `IFileHandle` were necessary for reasons I don't appreciate.

On the other hand, we might consider that this sort of idiom is something that is missing from the BCL -- if we consider `Stream` as analogous to `IEnumerator`, there is no analog in `System.IO` to `IEnumerable`. This is a problem I've encountered in multiple projects in .NET, that anything dealing with IO inevitably needs to invent a type whose job is, "please give me a stream when I ask for it" -- which is what `FileInfo` is, but that of course is very specific to files.

* `IFileHandle` (5X?)
* `SimpleFileHandle`

## Random

Category CN. This has been discussed in some issues so I will not repeat the discussion here. It seems like PRNGs are a sufficiently interesting topic that the BCL *might* benefit from a richer treatment than they have right now -- specifically in ML and other similar applications the benefit is on relatively weak but fast generators. Whereas you could imagine other applications in the other direction not being happy with Mersenne twisters or somesuch. Not sure about this.

* `IRandom` (4)
* `SysRandom`
* `TauswortheHybrid`, a really fast random number generator.
* `TauswortheHybrid.State`, a serializable "state" for a restartable object.
* `RandomUtils`, some static methods to do stuff with these.

## Useless

Category XXX.

* `ITrainerArguments`

Can be removed now. If I recall correctly this was necessary at one point due to being referenced in an object we wanted to deserialize, since our move away from using .NET serialization happened somewhat gradually. At this point though it appears to have no remaining usage. Oh the joys of .NET serialization.

## Legacy `ITrainer`, but still current `PredictionKind`

Category X, I think? In the process of being replaced with `ITrainerEstimator`, I think. However, `ITrainerEstimator` still has `PredictionKind`, which really ought to go away.

* `ITrainer`
* `ITrainer<out TPredictor>`
* (XXX) `PredictionKind`, an `enum`. Having in the central assembly an enumeration of all the types of possible predictions is anathematic to any sort of extensibility story.

## Signature Delegates

Category DXXX. Signatures used to serve two purposes: they told the dependency injection framework what "extra arguments" would be necessary to construct this object (a use that @eerhardt has replaced with `IComponentFactory` instances, thank you very much Eric.) For example, [see here](https://github.com/dotnet/machinelearning/blob/cde7038b3132801ee212b3757945a3697a1f485c/src/Microsoft.ML.StandardLearners/Standard/MultiClass/MetaMulticlassTrainer.cs#L26-L28). I'd much rather even have marker interfaces on the produced type, or subinterfaces of `IComponentFactory`, than keep these around.

Note that signature delegates appear in more places than this namespace, and the feedback should be considered universal.

* `SignatureAnomalyDetectorTrainer`
* `SignatureBinaryClassifierTrainer`
* `SignatureClusteringTrainer`
* `SignatureDefault`
* `SignatureMatrixRecommendingTrainer`
* `SignatureModelCombiner`
* `SignatureMultiClassClassifierTrainer`
* `SignatureMultiOutputRegressorTrainer`
* `SignatureRankerTrainer`
* `SignatureRegressorTrainer`
* `SignatureSequenceTrainer`
* `SignatureSuggestedSweepsParser`
* `SignatureSweeper`
* `SignatureSweepResultEvaluator`
* `SignatureTrainer`

## More Trainer

Category X? These are caught up with `ITrainer` but also `RoleMappedData`. The trouble is that I'm not sure how we'll deal with the same problems `RoleMappedData` was necessary to solve at the command line level.

* `TrainContext`
* `TrainerExtensions`
* `TrainerInfo`

# Namespace `Microsoft.ML.Core.Data`

This namespace is relatively new and contains foundational interfaces of the new estimator/transformer/data-view idiom [#581](https://github.com/dotnet/machinelearning/issues/581).

* `IDataReader<in TSource>` (1C)
* `IDataReaderEstimator<in TSource, out TReader>` (1C)
* `IEstimator<out TTransformer>` (1C)
* `ITransformer` (3)

The following are relevant to the "schema-lite" object used in estimators. It is somewhat analogous to `ISchema` but it can't quite be an `ISchema`.

* `SchemaShape`
* `SchemaShape.Column`
* `SchemaShape.Column.VectorKind`
* `SchemaException`

# Namespace `Microsoft.ML.Runtime.Command`

Category D. Moving this is somewhat tricky -- it should go wherever `ArgumentsAttribute` winds up. Also the utilities littering the command currently should be cleared up and moved somewhere else.

* `Microsoft.ML.Runtime.Command.ICommand`
* (X) `Microsoft.ML.Runtime.Command.SignatureCommand`

# Namespace `Microsoft.ML.Runtime.CommandLine`

Category D. The thing is that component authors will probably need *some* way to signal, "hey this is a configuration flag for entry points, the command line, the GUI, etc."

* `ArgumentAttribute`
* `ArgumentAttribute.VisibilityType`
* `ArgumentType`
* `SpecialPurpose`

Category DM. For the command line parser specifically, can just go live with MAML I think. There are one or two components that "assume" they're on the command line and still provide error messages and whatnot under that assumption.

* `CharCursor`
* `CmdLexer`
* `CmdParser`
* `CmdParser.ArgInfo`
* `CmdParser.ArgInfo.Arg`
* `CmdQuoter`
* `DefaultArgumentAttribute`
* `EnumValueDisplayAttribute`
* `ErrorReporter`
* `HideEnumValueAttribute`
* `ICommandLineComponentFactory`
* `SettingsFlags`

# Namespace `Microsoft.ML.Runtime.Data`

Category P. Most important namespace, probably, from a practical perspective. The home of `IDataView`.

* `IDataView` (3M), would have previously been (1) or (2) but I think the separation of concerns of what *had* been `IDataTransform` into `IEstimator`/`ITransformer`/`IDataView` may have made this possible. The only worry I have is, if we imagine this or something like this will be lifted into BCL, is there something about having that be an interface?
* `ICounted`, (2R), most basic interface on cursors describing their position.
* `ICursor`, (2RM), for describing *something* about movement, most important but not only descendant is `IRowCursor`.
* `IRow` (3MR), useful for operations over `IRowCursor` where we want to have some faith that the operator won't be messing with the cursor state, but also useful for things that just want to return a row of data, while using the `IDataView` idioms.
* `IRowCursor`, combination of `ICursor` and `IRow`. By far the most important implementation of either.
* `IRowCursorConsolidator` (5), used in cursor sets... but practically there is only one actual consolidator implementor, and given that the consolidator is meant to be applied to the *end* of a chain of cursors yet is instantiated at the *head*, it is unclear how multiple implementations would be even slightly helpful. We may even be able to stop returning it from here altogether.
* `IRowToRowMapper` (3M), very useful and more restrictive alternative to an `IDataView` -- basically an alternative to getting a cursor, used for more efficient row-by-row transformations than is possible (see for example speedup enabled by this via [#986](https://github.com/dotnet/machinelearning/issues/986)). Most `IDataView` *are* `IRowToRowMapper`, as a matter of fact. The presence and usefulness of this, including all the other things `IDataView` *might* do, is what makes me suspicious that actually turning it into a base class may not be possible.
* `ISchema` (4), for describing the dynamic data -- used practically everywhere to describe the shape of a "dataset."
* `ISchematized` (1M), used by many different things to indicate they can describe their shape... important implementors are data views, row mappers, rows, cursors, bound mappers.
* `CursorState`, an `enum` used to tell when a cursor hasn't started, is in process, has finished...
* `ValueGetter<TValue>`, delegate used to actually fetch values out of `IRow`, other similar things.

## Actual Types, and Type-Related Utilities

Special types used in `IDataView` pipelines.

Category P (mostly).

* `UInt128`, mostly for IDs.
* `VBuffer<T>`, for vector values.
* (I) `ReadOnlyMemoryUtils`, for operating over `ReadOnlyMemory<char>`, *mostly* intended as a temporary measure until the extensions for `ReadOnlyMember<char>` get shipped off to Framework.

## Column Types

Category P. The types as exposed by `ISchema` to describe data

* `BoolType`
* `ColumnType`
* `DataKind`
* `DataKindExtensions`
* `DateTimeOffsetType`
* `DateTimeType`
* `KeyType`
* `NumberType`
* `PrimitiveType`, any type whose values can be "independently copied" by assignment is one of these. All types in ML.NET *except* vectors and images.
* `StructuredType`, any type that can't do this is one of these. Vectors and image types.
* `TextType`
* `TimeSpanType`
* `VectorType`

## Auxiliary "Role Mapping" Information

Category PA, should become D I hope though the path is unclear.

The new API should make these unnecessary for *users* to know about I hope. The trouble is, these were built to solve specific problems that occur in command line and GUI (that problem being, how do you specify Features/Label/etc. columns *once* and have those settings apply to *many* learners, that stuff has to be put **somewhere**).

* `RoleMappedData`
* `RoleMappedSchema`
* `RoleMappedSchema.ColumnRole`

## For the implementation of `IDataView`

Category C. While `IDataView` itself is fairly expressive, there are a few common patterns, and for these we have some "helper" classes that can deal with these.

### Cursor Base Classes

Category CM, it seems like this could hypothetically live in `Data`?

* `LinkedRootCursorBase<TInput>`
* `LinkedRowFilterCursorBase`
* `LinkedRowRootCursorBase`
* `RootCursorBase`
* `SynchronizedCursorBase<TBase>`

## Important

Category C, though arguably *might* become P in some cases. While transformers can and sometimes do perform rather complex operations on data, 90% of the time it's something simple like mapping one input to one output, for which we have the following conveniences to simplify both consumption of data, and production of higher level abstractions like row mappers, and data views more generally.

* `IValueMapper` (2MR), mostly used by predictors that map one input to one output.
* `ValueMapper<TSrc, TDst>`, the delegate type mostly but not exclusively used by the above.
* `IValueMapperDist` (2M), mostly used by calibrated predictors to predict a distributional value.
* `ValueMapper<TVal1, TVal2, TVal3>`, the delegate type mostly but not exclusively used by the above.
* `RefPredicate<T>` is `bool(ref T)`.

## The Beloved Schema Bindable Mapper

Category X, I hope? Its fate is I suspect bound up with that of `RoleMappedData`, and will I *hope* become unnecessary with the introduction of the new `ITransformer` idioms. If `IValueMapper` and `IValueMapperDist` covers 90% of predictors, there's another 10% with more complex behavior.

* `ISchemaBindableMapper`
* `ISchemaBoundMapper`
* `ISchemaBoundRowMapper`

## For the implementation of `IHostEnvironment`

Category PMA (moved to `Data`, and also read [#1098](https://github.com/dotnet/machinelearning/issues/1098) to understand in what way these things will be public as this "becomes" `MLContext`) and also separately C, in the sense that the things related to `IHostEnvironment` will not be visible to the users.

* `ChannelProviderBase`
* `ChannelProviderBase.ExceptionContextKeys`
* `ConsoleEnvironment`
* `HostEnvironmentBase<TEnv>`
* `HostEnvironmentBase<TEnv>.HostBase`

## More HostEnvironmentStuff

Category C.

* (PX) `IMessageDispatcher`, maybe we can just say this functionity is just on `HostEnvironmentBase` rather than part of some separate interface.
* `IMessageSource` (3R), for pipes to describe themselves *without* also exposing their methods to dispatch more methods.

## Progress Reporting

Category IM. Move to `Data`. No reason for it to be public...

* `ProgressReporting`
* `ProgressReporting.ProgressChannel`
* `ProgressReporting.ProgressEntry`
* `ProgressReporting.ProgressEvent`
* `ProgressReporting.ProgressEvent.EventKind`
* `ProgressReporting.ProgressTracker`

## General Utility

Category I, maybe C.

* `ColumnInfo`, a really useful thing that holds a column name, index, and type. *Technically* an unnecessary type, since of course all that info is just in the schema -- but of course, when consuming a data view, 99% of the time it is helpful to know for a particular column its name *and* index *and* type, for which we have this utility.

## Consuming/Producing Metadata

Category PA in places, category C in others. The public parts are conveniences that make consuming "canonical" metadata a bit earlier. We ought to have some other way.

* `MetadataUtils`
* `MetadataUtils.Const`, holds the string constants of the canonical metadata.
* `MetadataUtils.Const.ScoreColumnKind`
* `MetadataUtils.Const.ScoreValueKind`
* `MetadataUtils.Kinds`
* `MetadataUtils.MetadataGetter<TValue>`


# Namespace `Microsoft.ML.Runtime.EntryPoints`

Generally I feel like this should go live somewhere else, *except* possibly `Optional` which has proven useful elsewhere it seems.

## Optional Values

Category PN. In ML applications, sometimes, often, it is most appropriate for how an algorithm is configured to depend on its inputs, using heuristics or other rules.

However, in these situations, we *also* sometimes We had to distinguish, most commonly but not exclusively among `string` types, the distinction between something that has a default value because the user didn't *touch* it.

So we need to distinguish between, "ok, are you `null` because no one touched you, in which case I should apply the default, or are you `null` because someone explicitly set you to `null`, in which case we are being told *not* to try to get this information?

These are structured similarly to `Nullable`, but used of course for a very different purpose and accepts (indeed, I think mostly only provides benefit to) reference types.

* `Optional`
* `Optional<T>`

## Entry-points Support

Category D, and M (I hope). I'm less familiar with this stuff, so I am not going to speculate on the fitness of the interfaces. Entry-points are basically command line on steroids.

* `EntryPointModuleAttribute`
* `EntryPointUtils`
* `IMlState`
* `IPredictorModel`
* `ITransformModel`
* `SignatureEntryPointModule`
* `TlcModule`
* `TlcModule.ComponentAttribute`
* `TlcModule.ComponentKindAttribute`
* `TlcModule.DataKind`
* `TlcModule.EntryPointAttribute`
* `TlcModule.EntryPointKindAttribute`
* `TlcModule.OptionalInputAttribute`
* `TlcModule.OutputAttribute`
* `TlcModule.RangeAttribute`
* `TlcModule.SweepableDiscreteParamAttribute`
* `TlcModule.SweepableFloatParamAttribute`
* `TlcModule.SweepableLongParamAttribute`
* `TlcModule.SweepableParamAttribute`

# Namespace `Microsoft.ML.Runtime.Internal.CpuMath`

Category X, *if* we're sure we never want to target anything pre-.NET Framework 4.6 going forward? The comments indicate this is a temporary utility to be used until a `System.Buffer.MemoryCopy` exists (which has always been in .NET Core but not .NET Framework until 4.6, and we wanted to continue working on 4.5 until fairly recently). Consists of a single class.

* `Microsoft.ML.Runtime.Internal.CpuMath.MemUtils`

# Namesapce `Microsoft.ML.Runtime.Internal.Utilities`

Category I, sometimes arguably C. The utilities are generally classes used multiple places in the codebase, but also some stuff some original authors *thought* might be useful, but turned out not to be. Also, collections of things that *were* perhaps generally useful in the past but as the library changed might have ceased to become useful. (So a fair amount of previously-lively-but-now-dead code is here.) Anything we *might* make public might be because it got lifted to the BCL.

## Widely Used but Specific to Machine Learning

Category I. Useful and generally widely used in our codebase, but so specific to ML that I don't really see they have much potential for use in any other applications.

Discretization. Used in FastTree and the binning-normalizer transformers. Discretization is useful for many reasons.

* `BinFinderBase`
* `DynamicBinFinder`
* `GreedyBinFinder`
* `SupervisedBinFinder`

Pass over a single streaming source to produce an in-memory sample. Really important in ML applications, not sure how useful it is elsewhere.

* `IReservoirSampler<T>` (5).
* `ReservoirSamplerWithoutReplacement<T>`
* `ReservoirSamplerWithReplacement<T>`

Statistics.

* `SummaryStatistics`, for calculating running statistics (up to the fourth moment if I remember correctly) using constant memory.
* `SummaryStatisticsBase`, same.
* `SummaryStatisticsUpToSecondOrderMoments`, same.

Others.

* `FixedSizeQueue<T>`, useful in n-grams, and other windowing operations.
* `Hashing`, widely used utility for produces hashes of stuff which is used in ML all the time for many reasons as we know.
* `Stats`, useful probabilistic functions.

## Silly Things

Things whose presence is at least moderately unacceptable.

* (X) `CharUtils`, what is wrong with `char.ToUpperInvariant` and `char.ToLowerInvariant`? As far as I see these methods have been in every version of .NET since .NET Framework 2.0.
* (M) `CmdIndenter`, all usages of this are in components that are assuming they're working off the command line. We should try to see if we can remove usages or, in the case of `TextSaver`, maybe consider moving this into the class itself. Or as part of whever command lines exist.
* (MI) `ExceptionMarshaller`, used exclusively in `BinaryLoader` and `BinarySaver`. Quite useful there but then maybe it should just be internal to the code there.
* (X *maybe* M) `HeapNode`, I'm not sure why this exists. It is a *separate* implementation of a priority queue to `Heap<T>` used in exactly one place. I'd rather that code use the existing one or, if somehow that existing implementation is insufficient, this odd one will move to be a private nested class of its only user.
* (X) `HeapNode.Heap<T>`, I think it would be nice if .NET `System.Collections.Generics` followed our example and put the implementation for what is currently `LinkedList<T>` inside `LinkedListNode<T>`.
* (XXX) `ListExtensions`, ridiculous.

## Future Deprecations

* `DoubleParser`, mostly only still here because Framework doesn't yet have `double.TryParse(ReadOnlySpan<char>, out double)`. Also Not sure about the speed of that method, should be investigated.
* `DoubleParser.Result`, similar.
* (A) `MathUtils`, ancient utility way overdue for cleanup, and whose ops are generally better expressed in `VectorUtils`. But still too widely used to annihilate.

## More Universal Stuff

Category IN.

* `BigArray<T>`, when 2 billion items isn't enough. (Surprisingly often in machine learning. :P )
* (X) `BigArray<T>.Visitor`, seems odd, not sure why we are encouraging this pattern.
* `Heap<T>`, it seems like .NET could benefit from something *like* [Java's priority queue](https://docs.oracle.com/javase/7/docs/api/java/util/PriorityQueue.html)? In many .NET projects I've seen just wind up implementing their own.
* `HybridMemoryStream`, the idea of a memory stream that once it reaches a certain size can spill over to disk storage seems like it is something that might be useful in many cases.
* `MadeObjectPool<T>`, I think you're aware of this.
* `ObjectPool<T>`, same.
* `ObjectPoolBase<T>`, same.
* `FloatUtils`, does a lot of float-wise operations, but probably most useful for getting bits out of numbers (useful for things like hashing). But many of these operations seem like good candidates for an outside library.
* `HashArray<TItem>`, the idea of an efficient "array" that maps indices to objects, and objects to indices, *and* can efficiently sort them seems to crop up a lot, since not just storing a set but also enumerating its members is super useful in lots of contexts, not just ML.
* (A) `NormStr`, while the idea of a centralized and *ordered* collection of strings is useful, I'm not sure I like how it was done here.
* (A) `NormStr.Pool`, same.

Category I.

* `IndentingTextWriter`
* `IndentingTextWriter.Scope`
* `LimitedConcurrencyLevelTaskScheduler`
* `LruCache<TKey, TValue>`

* `MatrixTransposeOps`, relatively fast transposers.
* `MinWaiter`, serialization primitive for when you have thread workers processing parallel but nonethleess ordered data, where you want them to "output" in a defined order.
* `OrderedWaiter`
* `PlatformUtils`
* `ResourceManagerUtils`
* `ResourceManagerUtils.ResourceDownloadResults`
* `SubsetStream`, useful for structured files involving blocks of data, where "sections" of the input stream should be processed as a unit.
* `TextReaderStream`, taking a text reader, presenting it as a stream.
* (M) `Tree<TKey, TValue>`, should probably move to just live with the server channel implementation. (Which is in TLC anyway, might want to move it over.)

* `Utils`, an incredibly large collection of grab-bag functions.

## Vector manipulation

Category I, *possibly* C. There are operations over `VBuffer`s that are useful enough, but that you nonetheless might hesitate to add to the class itself to keep from bloating it up.

* `VBufferUtils`
* `VBufferUtils.PairManipulator<TSrc, TDst>`
* `VBufferUtils.PairManipulatorCopy<TSrc, TDst>`
* `VBufferUtils.SlotValueManipulator<T>`
* `VBufferUtils.ValuePredicate<T>`

# Namespace `Microsoft.ML.Runtime.TreePredictor`

Category PXX, *maybe* M.

Internally there is a GUI tree ensemble visualizer. There are multiple tree learning packages that each produce tree ensembles, and rather than have the tree visualizer have to be aware of each and every one of them and have references to all of them, the idea is that all these disparate packages will have their objects implement this common interface. In this way, the code for the tree visualizer itself is simplified.

That by itself is not a ridiculous idea. However, practically only FastTree and LightGBM (which share their basic model format) see significant usage. Also this interface set is only really useful if multiple packages can reference it, which is somewhat difficult to do with it lying in `Microsoft.ML.Core`. In any case even if we somehow thought the idea was worth preserving, it *definitely* does not belong in `Core`.

* `Microsoft.ML.Runtime.TreePredictor.INode` (1M)
* `Microsoft.ML.Runtime.TreePredictor.ITree` (1M)
* `Microsoft.ML.Runtime.TreePredictor.ITree<TFeatures>`
* `Microsoft.ML.Runtime.TreePredictor.ITreeEnsemble` (1M)
* `Microsoft.ML.Runtime.TreePredictor.NodeKeys`
