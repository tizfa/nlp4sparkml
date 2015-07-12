/*
 * *****************
 *  Copyright 2015 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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
 * *******************
 */

package it.cnr.isti.hlt.nlp4sparkml.indexer;

import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.UID;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.Param;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Created by Tiziano on 11/07/2015.
 */
public class IdentifierIndexerModel extends Model<IdentifierIndexerModel> {

    private final String uid;
    private final JavaPairRDD<String, Long> featuresMappingRDD;

    private final Param<List<String>> inputCols;
    private final Param<String> outputCol;

    public IdentifierIndexerModel(JavaPairRDD<String, Long> featuresMapping) {
        Cond.requireNotNull(featuresMapping, "featuresMappingRDD");
        this.uid = UID.generateUID(getClass());
        this.inputCols = new Param<List<String>>(this, "inputColumns", "The set of input columns to be indexed");
        setDefault(this.inputCols, new ArrayList<String>());
        this.outputCol = new Param<String>(this, "outputCol", "The output column containing features indexed");
        setDefault(this.outputCol, "featuresIndexed");
        this.featuresMappingRDD = featuresMapping;
    }

    public Param<List<String>> inputCol() {
        return inputCols;
    }

    public Param<String> outputCol() {
        return outputCol;
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
        JavaRDD<Row> indexedFeatures = dataset.toJavaRDD().map(row -> {
            HashMap<String, Long> mapFeatures = new HashMap<String, Long>();
            ArrayList<Long> featuresIndexed = new ArrayList<Long>();
            for (int i = 0; i < row.length(); i++) {
                List<String> features = row.getList(i);
                for (String feature : features) {
                    if (mapFeatures.containsKey(feature))
                        featuresIndexed.add(mapFeatures.get(feature));
                    else {
                        List<Long> id = featuresMappingRDD.lookup(feature);
                        if (id.size() > 0) {
                            long idFeature = id.get(0);
                            mapFeatures.put(feature, idFeature);
                            featuresIndexed.add(idFeature);
                        }
                    }
                }
            }
            Object[] values = new Object[row.length()+1];
            for (int i = 0; i < row.length();i++)
                values[i] = row.get(i);
            values[values.length-1] = featuresIndexed;
            return RowFactory.create(values);
        });

        StructType newSchema = transformSchema(dataset.schema());
        return dataset.sqlContext().createDataFrame(indexedFeatures, newSchema);
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

    @Override
    public String uid() {
        return uid;
    }
}
