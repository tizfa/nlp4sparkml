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

package it.cnr.isti.hlt.nlp4sparkml.classifier;

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.data.MultilabelPoint;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.UID;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.Param;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;

/**
 * A generic multilabel multiclass classifier in the form of a Spark ML estimator.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class MultilabelClassifierEstimator<T extends Model<T>> extends Estimator<T> {

    private final Param<String> inputCol = new Param<String>(this, "inputCol", "Input column name");
    private final Param<String> outputCol = new Param<String>(this, "outputCol", "Output column name");

    /**
     * Get the input column name.
     *
     * @return The input column name.
     */
    public String getInputCol() {
        return getOrDefault(inputCol);
    }

    public MultilabelClassifierEstimator<T> setInputCol(String inputCol) {
        set(this.inputCol, inputCol);
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

    public MultilabelClassifierEstimator<T> setOutputCol(String outputCol) {
        set(this.outputCol, outputCol);
        return this;
    }

    /**
     * Declare all the necessary broadcast variables. The subclasses need to store the broadcast variables somewhere
     * at its internal.
     *
     * @param sc The spark context.
     */
    protected abstract void initBroadcastVariables(JavaSparkContext sc);


    /**
     * Destroy the the broadcast variables previously defined with {@link #initBroadcastVariables(JavaSparkContext)}.
     */
    protected abstract void destroyBroadcastVariables();

    @Override
    public T fit(DataFrame dataset) {
        Cond.requireNotNull(dataset, "dataset");
        StructType updatedSchema = transformSchema(dataset.schema());
        DataFrame df = dataset.withColumn(getOutputCol(), dataset.col(getInputCol()));
        JavaRDD<Row> rows = df.javaRDD();
        int nf = DataUtils.computeNumFeaturesFromDataFrame(rows, getInputCol());
        JavaSparkContext sc = new JavaSparkContext(rows.context());
        Broadcast<Integer> numFeatures = sc.broadcast(nf);
        JavaRDD<MultilabelPoint> inputPoints = rows.map(row -> {
            int inIndex = row.fieldIndex(getInputCol());
            int outIndex = row.fieldIndex(getOutputCol());
            Row inputPoint = row.getStruct(inIndex);
            MultilabelPoint pt = DataUtils.toMultilabelPoint(row, getInputCol(), numFeatures.value());
            return pt;
        });

        initBroadcastVariables(sc);
        T model = buildClassifier(inputPoints, nf);
        destroyBroadcastVariables();
        return model;
    }

    /**
     * Build the specific learning model by processing input training points available in the specified RDD.
     *
     * @param inputPoints The set of input points used as training data.
     * @param numFeatures The total number of features available in training dataset.
     * @return The corresponding classifier.
     */
    protected abstract T buildClassifier(JavaRDD<MultilabelPoint> inputPoints, int numFeatures);


    @Override
    public StructType transformSchema(StructType structType) {
        // TODO In an estimator not sure which is the purpose of this method. Until now, return
        // the same schema as input schema.

        /*DataType inputType = structType.apply(getInputCol()).dataType();
        this.validateInputType(inputType);
        List<String> names = Arrays.asList(structType.fieldNames());
        Cond.require(!names.contains(getOutputCol()), "The output column " + getOutputCol() + " already exists in this schema!");
        List<StructField> fields = new ArrayList<>();
        for (int i = 0; i < structType.fields().length; i++) {
            fields.add(structType.fields()[i]);
        }
        DataType outDataType = DataUtils.pointClassificationResultsDataType();
        fields.add(DataTypes.createStructField(getOutputCol(), outDataType, false));
        return DataTypes.createStructType(fields);*/
        return structType;
    }

    protected void validateInputType(DataType inputType) {
        Cond.requireNotNull(inputType, "inputType");
        Cond.require(inputType instanceof StructType, "The type of 'inputType' parameter must be 'StructType'");
        DataUtils.checkMultilabelPointDataType((StructType) inputType);
    }

    @Override
    public String uid() {
        return UID.generateUID(getClass());
    }
}
