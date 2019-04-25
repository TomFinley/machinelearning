// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// A set of 'requirements' to the incoming schema, as well as a set of 'promises' of the outgoing schema. This is
    /// more relaxed than the proper <see cref="DataViewSchema"/>, since it's only a subset of the columns, and also
    /// since it doesn't specify exact <see cref="DataViewType"/>'s for vectors and keys.
    /// </summary>
    /// <remarks>
    /// Since this object's purpose is to communicate information about what schemas should broadly look like ahead of
    /// the work of creating a <see cref="ITransformer"/>, descriptions of this class will center around how they relate
    /// to the sort of <see cref="DataViewSchema"/> they should correspond to. Specifically, for every
    /// <see cref="DataViewSchema.Column"/> in a <see cref="DataViewSchema"/> where
    /// <see cref="DataViewSchema.Column.IsHidden"/> is <see langword="false"/>, there should be a
    /// <see cref="SchemaShape.Column"/> that corresponds to it in terms of being descriptive with regard to its name,
    /// type information, and descriptive with regard to its annotations.
    /// </remarks>
    /// <seealso cref="IEstimator{TTransformer}.GetOutputSchema(SchemaShape)"/>
    public sealed class SchemaShape : IReadOnlyList<SchemaShape.Column>
    {
        private readonly Column[] _columns;

        private static readonly SchemaShape _empty = new SchemaShape(Enumerable.Empty<Column>());

        /// <summary>
        /// The number of columns.
        /// </summary>
        public int Count => _columns.Length;

        public Column this[int index] => _columns[index];

        /// <summary>
        /// Each of these correspond to a non-hidden <see cref="DataViewSchema.Column"/>, telling the name of the
        /// column, indications as to the general sort of type, and what annotations that column should have.
        /// </summary>
        /// <remarks>
        /// To give a practical example, to describe the "shape" of a <see cref="DataViewSchema.Column"/> with the type
        /// <see cref="VectorDataViewType"/> of known-size with item-type <see cref="NumberDataViewType.Single"/>,
        /// <see cref="SchemaShape.Column.ItemType"/> would be <see cref="NumberDataViewType.Single"/>,
        /// <see cref="IsKey"/> would be <see langword="false"/>, and <see cref="Kind"/> would be
        /// <see cref="VectorKind.Vector"/>.
        /// </remarks>
        public readonly struct Column
        {
            /// <summary>
            /// This enumeration describes what sort of <see cref="VectorDataViewType"/> is being described,
            /// or indeed if there is a data view type at all.
            /// </summary>
            public enum VectorKind
            {
                /// <summary>
                /// In this case, this column is not describing a <see cref="VectorDataViewType"/> at all.
                /// </summary>
                Scalar,
                /// <summary>
                /// In this case, the <see cref="DataViewSchema.Column.Type"/> this corresponds to is a
                /// <see cref="VectorDataViewType"/> where <see cref="VectorDataViewType.IsKnownSize"/>
                /// is <see langword="true"/>.
                /// </summary>
                Vector,
                /// <summary>
                /// In this case, the <see cref="DataViewSchema.Column.Type"/> this corresponds to is a
                /// <see cref="VectorDataViewType"/> where <see cref="VectorDataViewType.IsKnownSize"/>
                /// is <see langword="false"/>.
                /// </summary>
                VariableVector
            }

            /// <summary>
            /// The column name. The <see cref="DataViewSchema.Column"/> to which this corresponds must have precisely
            /// this name.
            /// </summary>
            public readonly string Name;

            /// <summary>
            /// The type of the column: scalar, fixed vector or variable vector.
            /// </summary>
            public readonly VectorKind Kind;

            /// <summary>
            /// The 'raw' type of column item.
            /// </summary>
            /// <remarks>
            /// This entry is meant to be descriptive as regards the type in <see cref="DataViewSchema.Column.Type"/>, but is
            /// will not necessarily have the same value.
            ///
            /// For <see cref="IEstimator{TTransformer}"/> implementations that deal with "custom" <see cref="DataViewType"/>
            /// derived types, that is, those types that are not defined in the same assembly as <see cref="IDataView"/>,
            /// the <see cref="IEstimator{TTransformer}"/> and <see cref="ITransformer"/> implementations are free to develop
            /// whatever conventions are appropriate for their usage.
            ///
            /// For the standard types, aside from <see cref="VectorDataViewType"/> and <see cref="KeyDataViewType"/>,
            /// this field will hold the same value as <see cref="DataViewSchema.Column.Type"/> as the corresponding type.
            /// </remarks>
            public readonly DataViewType ItemType;

            /// <summary>
            /// The flag whether the column is actually a key. If <see langword="true"/>, <see cref="ItemType"/> is
            /// representing the underlying primitive type, that is, one of <see cref="NumberDataViewType.Byte"/>,
            /// <see cref="NumberDataViewType.UInt16"/>, <see cref="NumberDataViewType.UInt32"/>, or
            /// <see cref="NumberDataViewType.UInt64"/>, with the actual item type in the corresponding columns of
            /// <see cref="KeyDataViewType"/> with the <see cref="DataViewType.RawType"/> correspdoning to one of those
            /// unsigned types.
            /// </summary>
            public readonly bool IsKey;

            /// <summary>
            /// The annotations that are present for this column.
            /// </summary>
            public readonly SchemaShape Annotations;

            /// <summary>
            /// Defines a column for a schema shape.
            /// </summary>
            /// <remarks>
            /// In similar form to how <see cref="SchemaShape"/> is considered a "promise" for a <see cref="DataViewSchema"/> yet to come,
            /// so too is this considered a "promise" for a <see cref="DataViewSchema.Column"/> yet to come. The structures defined here
            /// are therefore phrased in terms of the underlying "promise" they are making.
            /// </remarks>
            /// <param name="name">The name of the column being described.</param>
            /// <param name="vecKind">The "kind" of vector-arity, whether it be scalar, known size, or unknown size vector.</param>
            /// <param name="itemType">The type of the item.</param>
            /// <param name="isKey">Whether the item type is supposed to be of key.</param>
            /// <param name="annotations">A description of the annotations for this column.</param>
            [BestFriend]
            internal Column(string name, VectorKind vecKind, DataViewType itemType, bool isKey, SchemaShape annotations = null)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValueOrNull(annotations);
                Contracts.CheckParam(!(itemType is KeyDataViewType), nameof(itemType), "Item type cannot be a key");
                Contracts.CheckParam(!(itemType is VectorDataViewType), nameof(itemType), "Item type cannot be a vector");
                Contracts.CheckParam(!isKey || KeyDataViewType.IsValidDataType(itemType.RawType), nameof(itemType), "The item type must be valid for a key");
                Contracts.CheckParam(vecKind == VectorKind.Scalar || itemType is PrimitiveDataViewType, nameof(itemType),
                    "If " + nameof(vecKind) + " is not " + nameof(VectorKind.Scalar) + ", then the item type must be a " + nameof(PrimitiveDataViewType));

                Name = name;
                Kind = vecKind;
                ItemType = itemType;
                IsKey = isKey;
                Annotations = annotations ?? _empty;
            }

            /// <summary>
            /// Returns whether <paramref name="source"/> is a valid input, if this object represents a
            /// requirement.
            ///
            /// Namely, it returns true if and only if:
            ///  - The <see cref="Name"/>, <see cref="Kind"/>, <see cref="ItemType"/>, <see cref="IsKey"/> fields match.
            ///  - The columns of <see cref="Annotations"/> of <paramref name="source"/> is a superset of our <see cref="Annotations"/> columns.
            ///  - Each such annotation column is itself compatible with the input annotation column.
            /// </summary>
            [BestFriend]
            internal bool IsCompatibleWith(Column source)
            {
                Contracts.Check(source.IsValid, nameof(source));
                if (Name != source.Name)
                    return false;
                if (Kind != source.Kind)
                    return false;
                if (!ItemType.Equals(source.ItemType))
                    return false;
                if (IsKey != source.IsKey)
                    return false;
                foreach (var annotationCol in Annotations)
                {
                    if (!source.Annotations.TryFindColumn(annotationCol.Name, out var inputAnnotationCol))
                        return false;
                    if (!annotationCol.IsCompatibleWith(inputAnnotationCol))
                        return false;
                }
                return true;
            }

            [BestFriend]
            internal string GetTypeString()
            {
                string result = ItemType.ToString();
                if (IsKey)
                    result = $"Key<{result}>";
                if (Kind == VectorKind.Vector)
                    result = $"Vector<{result}>";
                else if (Kind == VectorKind.VariableVector)
                    result = $"VarVector<{result}>";
                return result;
            }

            /// <summary>
            /// Return if this structure is not identical to the default value of <see cref="Column"/>. If true,
            /// it means this structure is initialized properly and therefore considered as valid.
            /// </summary>
            [BestFriend]
            internal bool IsValid => Name != null;
        }

        public SchemaShape(IEnumerable<Column> columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            _columns = columns.ToArray();
            Contracts.CheckParam(columns.All(c => c.IsValid), nameof(columns), "Some items are not initialized properly.");
        }

        /// <summary>
        /// Given a <paramref name="type"/>, extract the type parameters that describe this type
        /// as a <see cref="SchemaShape"/>'s column type.
        /// </summary>
        /// <param name="type">The actual column type to process.</param>
        /// <param name="vecKind">The vector kind of <paramref name="type"/>.</param>
        /// <param name="itemType">The item type of <paramref name="type"/>.</param>
        /// <param name="isKey">Whether <paramref name="type"/> (or its item type) is a key.</param>
        [BestFriend]
        internal static void GetColumnTypeShape(DataViewType type,
            out Column.VectorKind vecKind,
            out DataViewType itemType,
            out bool isKey)
        {
            if (type is VectorDataViewType vectorType)
            {
                if (vectorType.IsKnownSize)
                {
                    vecKind = Column.VectorKind.Vector;
                }
                else
                {
                    vecKind = Column.VectorKind.VariableVector;
                }

                itemType = vectorType.ItemType;
            }
            else
            {
                vecKind = Column.VectorKind.Scalar;
                itemType = type;
            }

            isKey = itemType is KeyDataViewType;
            if (isKey)
                itemType = ColumnTypeExtensions.PrimitiveTypeFromType(itemType.RawType);
        }

        /// <summary>
        /// Create a schema shape out of the fully defined schema.
        /// </summary>
        [BestFriend]
        internal static SchemaShape Create(DataViewSchema schema)
        {
            Contracts.CheckValue(schema, nameof(schema));
            var cols = new List<Column>();

            for (int iCol = 0; iCol < schema.Count; iCol++)
            {
                if (!schema[iCol].IsHidden)
                {
                    // First create the annotations.
                    var mCols = new List<Column>();
                    foreach (var annotationColumn in schema[iCol].Annotations.Schema)
                    {
                        GetColumnTypeShape(annotationColumn.Type, out var mVecKind, out var mItemType, out var mIsKey);
                        mCols.Add(new Column(annotationColumn.Name, mVecKind, mItemType, mIsKey));
                    }
                    var annotations = mCols.Count > 0 ? new SchemaShape(mCols) : _empty;
                    // Next create the single column.
                    GetColumnTypeShape(schema[iCol].Type, out var vecKind, out var itemType, out var isKey);
                    cols.Add(new Column(schema[iCol].Name, vecKind, itemType, isKey, annotations));
                }
            }
            return new SchemaShape(cols);
        }

        /// <summary>
        /// Returns if there is a column with a specified <paramref name="name"/> and if so stores it in <paramref name="column"/>.
        /// </summary>
        [BestFriend]
        internal bool TryFindColumn(string name, out Column column)
        {
            Contracts.CheckValue(name, nameof(name));
            column = _columns.FirstOrDefault(x => x.Name == name);
            return column.IsValid;
        }

        public IEnumerator<Column> GetEnumerator() => ((IEnumerable<Column>)_columns).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        // REVIEW: I think we should have an IsCompatible method to check if it's OK to use one schema shape
        // as an input to another schema shape. I started writing, but realized that there's more than one way to check for
        // the 'compatibility': as in, 'CAN be compatible' vs. 'WILL be compatible'.
    }

    /// <summary>
    /// The 'data loader' takes a certain kind of input and turns it into an <see cref="IDataView"/>.
    /// </summary>
    /// <typeparam name="TSource">The type of input the loader takes.</typeparam>
    public interface IDataLoader<in TSource> : ICanSaveModel
    {
        /// <summary>
        /// Produce the data view from the specified input.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        IDataView Load(TSource input);

        /// <summary>
        /// The output schema of the loader.
        /// </summary>
        DataViewSchema GetOutputSchema();
    }

    /// <summary>
    /// Sometimes we need to 'fit' an <see cref="IDataLoader{TIn}"/>.
    /// A DataLoader estimator is the object that does it.
    /// </summary>
    public interface IDataLoaderEstimator<in TSource, out TLoader>
        where TLoader : IDataLoader<TSource>
    {
        // REVIEW: you could consider the transformer to take a different <typeparamref name="TSource"/>, but we don't have such components
        // yet, so why complicate matters?

        /// <summary>
        /// Train and return a data loader.
        /// </summary>
        TLoader Fit(TSource input);

        /// <summary>
        /// The 'promise' of the output schema.
        /// It will be used for schema propagation.
        /// </summary>
        SchemaShape GetOutputSchema();
    }

    /// <summary>
    /// The transformer is a component that transforms data.
    /// It also supports 'schema propagation' to answer the question of 'how will the data with this schema look, after you transform it?'.
    /// </summary>
    public interface ITransformer : ICanSaveModel
    {
        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        /// <remarks>
        /// One of the key requirements of <see cref="ITransformer"/> is that if one were to call <see cref="Transform(IDataView)"/>,
        /// the <see cref="IDataView.Schema"/> from the return value should be indistinguishable (barring object references, naturally)
        /// from the return value of this function.
        /// </remarks>
        DataViewSchema GetOutputSchema(DataViewSchema inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so in most implementations the actual transformations happens when
        /// one opens a cursor using <see cref="IDataView.GetRowCursor(IEnumerable{DataViewSchema.Column}, System.Random)"/>.
        /// </summary>
        /// <param name="input">The input to transform.</param>
        /// <returns>The output of the transformation.</returns>
        IDataView Transform(IDataView input);

        /// <summary>
        /// Whether a call to <see cref="GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool IsRowToRowMapper { get; }

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema. If <see cref="IsRowToRowMapper"/>
        /// is <c>false</c>, then an exception should be thrown. If the input schema is in any way
        /// unsuitable for constructing the mapper, an exception should likewise be thrown.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema);
    }

    /// <summary>
    /// The estimator (in Spark terminology) is an 'untrained transformer'. It needs to 'fit' on the data to manufacture
    /// a transformer.
    /// It also provides the 'schema propagation' like transformers do, but over <see cref="SchemaShape"/> instead of <see cref="DataViewSchema"/>.
    /// </summary>
    public interface IEstimator<out TTransformer>
        where TTransformer : ITransformer
    {
        /// <summary>
        /// Train and return a transformer.
        /// </summary>
        TTransformer Fit(IDataView input);

        /// <summary>
        /// Schema propagation for estimators. Returns the output schema shape of the estimator, if the input schema
        /// shape is like the one provided.
        /// </summary>
        /// <remarks>
        /// The purpose of this method is to perform a preliminary form of schema validation and propagation, prior to
        /// the more rigorous validation and propagation performed by
        /// <see cref="IEstimator{TTransformer}.GetOutputSchema(SchemaShape)"/>. The <see cref="ITransformer"/> is
        /// available only after calling <see cref="Fit(IDataView)"/>, which can take a considerable amount of time,
        /// often involving one or more passes over the dataset.
        ///
        /// So, to give an example, when you have an <see cref="IEstimator{TTransformer}"/> for bag-of-words encoding of
        /// text, to perform the tokenization, build dictionaries, construct the feature names, that requires at least
        /// one pass over the dataset in addition to taking a fair amount of memory. Yet, before fitting, you know the
        /// result of a transformation will be a column with a name, that this column will be a vector of <see
        /// cref="float"/> of known size (even though, prior to fitting, we do not yet know what that size will be), and
        /// we also know that there will be <c>SlotNames</c> annotation for this column (even though, again, we don't
        /// know what those names will be).
        ///
        /// Even in that indefinite form, this can catch many common errors. For example, simple typos in the column
        /// names, basic misunderstandings about what type is acceptable to what components, are one of the primary goals
        /// of this structure.
        ///
        /// The general expectation with <see cref="SchemaShape"/> being a "relaxed" schema means that not all the
        /// validation that may happen during <see cref="Fit(IDataView)"/> or the <see cref="ITransformer"/> methods is
        /// possible to do in this method. That is, this method offers a somewhat relaxed form of validation. The goal
        /// of implementors should be, if the methods on a <see cref="ITransformer"/> returned from
        /// <see cref="Fit(IDataView)"/> would succeed, then so too should this validation succeed.
        ///
        /// There is a small and usually unimportant detail about the correspondence between <see cref="DataViewSchema.Annotations"/>
        /// and <see cref="SchemaShape.Column.Annotations"/>. One of the design implications of
        /// annotations being optional and defined after the fact is that if an annotation is not of the "right" form, it
        /// is most correct for the <see cref="ITransformer"/> instance to treat it as if it does not have that annotation.
        /// Due to the relaxed nature of <see cref="SchemaShape"/>, it sometimes chances that it will look like
        /// there will be an annotation that is present, but that it becomes clear upon inspection of the actual annotation
        /// that the annotation is incorrect in some fashion, that will lead us to behave as if that annotation was not
        /// there. To give a practical example, we might see a <c>SlotNames</c> annotation on the
        /// <see cref="SchemaShape.Column.Annotations"/>, but then once we get the actual <see cref="DataViewSchema.Annotations"/>
        /// structure we see it is not of a compatible size to the column itself, or some other disqualifying factor. But,
        /// for various other reasons described elsewhere, we cannot consider an annotation being of unexpected form as being
        /// an error. So, in that case, the <see cref="ITransformer"/>'s schema propagation might begin to differ from the
        /// <see cref="IEstimator{TTransformer}"/>'s schema propagation, with regard to the annotations. However, this condition
        /// depends on an annotation existing outside of the expected range. All ML.NET components provide annotations inside
        /// the expected range, so the only way this situation could arise is if a user implemented <see cref="IDataView"/>
        /// goes out of its way to provide what amounts to a "malformed" annotation.
        /// </remarks>
        /// <returns>
        /// The <see cref="SchemaShape"/> that would correspond to the <see cref="DataViewSchema"/> returned from
        /// <see cref="ITransformer.GetOutputSchema(DataViewSchema)"/>, if <paramref name="inputSchema"/> corresponded
        /// to the <see cref="DataViewSchema"/> input into that method.
        /// </returns>
        SchemaShape GetOutputSchema(SchemaShape inputSchema);
    }
}
