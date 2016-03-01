/*
 *
 * ****************
 * Copyright 2015 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ******************
 */

package it.cnr.isti.hlt.nlp4sparkml.indexer;

import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.JavaModel;
import it.cnr.isti.hlt.nlp4sparkml.utils.UID;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.Param;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;

/**
 * Created by Tiziano on 11/07/2015.
 */
public class IdentifierIndexerModel extends JavaModel<IdentifierIndexerModel> {

    private final DataFrame featuresMapping;
    private final long numDistinctFeatures;

    private final Param<String> idCol;
    private final Param<List<String>> inputCols;
    private final Param<String> outputCol;

    public IdentifierIndexerModel(DataFrame featuresMapping, long numDistinctFeatures) {
        Cond.requireNotNull(featuresMapping, "featuresMapping");
        Cond.require(numDistinctFeatures > 0, "The number of distinct features must be greater than 0");
        this.idCol = new Param<String>(this, "idCol", "The column which unique identifies a single row");
        setDefault(this.idCol, "id");
        this.inputCols = new Param<List<String>>(this, "inputColumns", "The set of input columns to be indexed");
        setDefault(this.inputCols, new ArrayList<String>());
        this.outputCol = new Param<String>(this, "outputCol", "The output column containing features indexed");
        setDefault(this.outputCol, "featuresIndexed");
        this.featuresMapping = featuresMapping;
        this.numDistinctFeatures = numDistinctFeatures;
    }

    public long getNumDistinctFeatures() {
        return numDistinctFeatures;
    }

    public Param<String> idCol() {
        return idCol;
    }

    public Param<List<String>> inputCol() {
        return inputCols;
    }

    public Param<String> outputCol() {
        return outputCol;
    }


    public DataFrame getInternalFeaturesMappinng() {
        return featuresMapping;
    }

    /**
     * Get the id column name.
     *
     * @return The id column name.
     */
    public String getIdCol() {
        return getOrDefault(idCol);
    }

    public IdentifierIndexerModel setIdCol(String idCol) {
        set(this.idCol, idCol);
        return this;
    }

    /**
     * Get set of input column names.
     *
     * @return The set of input column names.
     */
    public List<String> getInputCols() {
        return getOrDefault(inputCols);
    }


    public IdentifierIndexerModel setInputCol(List<String> inputCols) {
        Cond.requireNotNull(inputCols, "inputCols");
        Cond.require(inputCols.size() > 0, "The set of input columns must not be empty!");
        set(this.inputCols, inputCols);
        return this;
    }

    /**
     * Get the output column name.
     *
     * @return The output column name.
     */
    public String getOutputCol() {
        return getOrDefault(outputCol);
    }

    public IdentifierIndexerModel setOutputCol(String outputCol) {
        set(this.outputCol, outputCol);
        return this;
    }


    @Override
    public DataFrame transform(DataFrame dataset) {
        Cond.requireNotNull(dataset, "dataset");
        ArrayList<String> fieldsToAnalyze = new ArrayList<>(getInputCols());
        JavaSparkContext sc = new JavaSparkContext(dataset.sqlContext().sparkContext());
        Broadcast<ArrayList<String>> inputFields = sc.broadcast(fieldsToAnalyze);
        String idCol = getIdCol();
        String outputCol = getOutputCol();

        // Create a temporary data frame containing the expansion of all original features
        // and keeping track of the original row ID.
        JavaRDD<Row> indexedFeatures = dataset.toJavaRDD().flatMap(row -> {
            long idRow = row.getLong(row.fieldIndex(idCol));
            List<String> inputFieldsLocal = inputFields.value();
            ArrayList<Tuple2<String, Long>> values = new ArrayList<Tuple2<String, Long>>();
            for (int i = 0; i < inputFieldsLocal.size(); i++) {
                List<String> features = row.getList(row.fieldIndex(inputFieldsLocal.get(i)));
                for (String feature : features) {
                    values.add(new Tuple2<>(feature, idRow));
                }
            }
            return values;
        }).map(pair -> {
            return RowFactory.create(pair._1(), pair._2());
        });
        StructType tmpSchema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(IdentifierIndexer.FEATURE, DataTypes.StringType, false),
                DataTypes.createStructField(getIdCol(), DataTypes.LongType, false)});
        DataFrame tmpDF = dataset.sqlContext().createDataFrame(indexedFeatures, tmpSchema).persist(StorageLevel.MEMORY_AND_DISK());

        // Join input rows with features IDs.
        DataFrame joinedRes = tmpDF.join(featuresMapping, tmpDF.col(IdentifierIndexer.FEATURE).equalTo(featuresMapping.col(IdentifierIndexer.FEATURE))).select(tmpDF.col(getIdCol()), featuresMapping.col(IdentifierIndexer.ID_FEATURE));

        // Generate a new dataframe by recombining all features of each specific row index.
        JavaRDD<Row> indexedRows = joinedRes.toJavaRDD().mapToPair(row -> {
            long idRow = row.getLong(0);
            long idFeature = row.getLong(1);
            return new Tuple2<Long, Long>(idRow, idFeature);
        }).groupByKey().map(v -> {
            Iterator<Long> featuresIDs = v._2().iterator();
            ArrayList<Long> feats = new ArrayList<Long>();
            while (featuresIDs.hasNext()) {
                feats.add(featuresIDs.next());
            }
            Long[] featsArray = feats.toArray(new Long[0]);
            return RowFactory.create(v._1(), featsArray);
        }).persist(StorageLevel.MEMORY_AND_DISK());
        StructType indexedSchema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(idCol, DataTypes.LongType, false),
                DataTypes.createStructField(outputCol, DataTypes.createArrayType(DataTypes.LongType), false)});
        DataFrame indexedDF = dataset.sqlContext().createDataFrame(indexedRows, indexedSchema).persist(StorageLevel.MEMORY_AND_DISK());

        // Merge the original dataset and the generated dataframe with the features indexed.
        String[] columnsOriginal = dataset.columns();
        Column[] cols = new Column[columnsOriginal.length + 1];
        for (int i = 0; i < columnsOriginal.length; i++)
            cols[i] = dataset.col(columnsOriginal[i]);
        cols[cols.length - 1] = indexedDF.col(outputCol);
        DataFrame res = dataset.join(indexedDF, dataset.col(getIdCol()).equalTo(indexedDF.col(getIdCol()))).select(cols);
        return res;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        Cond.requireNotNull(schema, "schema");
        List<String> names = Arrays.asList(schema.fieldNames());
        Cond.require(!names.contains(getOutputCol()), "The output column " + getOutputCol() + " already exists in this schema!");
        List<StructField> fields = new ArrayList<>();
        for (int i = 0; i < schema.fields().length; i++) {
            fields.add(schema.fields()[i]);
        }
        DataType dt = DataTypes.createArrayType(DataTypes.LongType);
        fields.add(DataTypes.createStructField(getOutputCol(), dt, false));
        return DataTypes.createStructType(fields);
    }

}
