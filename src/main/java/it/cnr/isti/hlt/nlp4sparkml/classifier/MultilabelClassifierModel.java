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
import it.cnr.isti.hlt.nlp4sparkml.data.PointClassificationResults;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.UID;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
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
import java.util.List;

/**
 * An abstract multilabel classifier in the form of Spark ML model.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class MultilabelClassifierModel<T extends MultilabelClassifierModel<T>> extends Model<T> {

    private final Param<String> inputCol = new Param<String>(this, "inputCol", "Input column name");
    private final Param<String> outputCol = new Param<String>(this, "outputCol", "Output column name");

    /**
     * The total number of features used to build the model.
     */
    private final int nf;

    public MultilabelClassifierModel(int numFeatures) {
        Cond.require(numFeatures > 0, "The number of features is less than 1");
        this.nf = numFeatures;
    }

    /**
     * Get the input column name.
     *
     * @return The input column name.
     */
    public String getInputCol() {
        return getOrDefault(inputCol);
    }

    public MultilabelClassifierModel setInputCol(String inputCol) {
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

    public MultilabelClassifierModel setOutputCol(String outputCol) {
        set(this.outputCol, outputCol);
        return this;
    }


    @Override
    public DataFrame transform(DataFrame dataset) {
        Cond.requireNotNull(dataset, "dataset");
        StructType updatedSchema = transformSchema(dataset.schema());
        DataFrame df = dataset.withColumn(getOutputCol(), dataset.col(getInputCol()));
        JavaRDD<Row> rows = df.javaRDD();
        JavaSparkContext sc = new JavaSparkContext(rows.context());
        Broadcast<Integer> numFeatures = sc.broadcast(nf);
        initBroadcastVariables(sc);
        JavaRDD<Row> updatedRows = rows.map(row -> {
            int outIndex = row.fieldIndex(getOutputCol());
            MultilabelPoint pt = DataUtils.toMultilabelPoint(row, getInputCol(), numFeatures.value());
            PointClassificationResults res = classifyPoint(pt);
            Object[] values = new Object[row.size()];
            for (int i = 0; i < row.size(); i++) {
                if (i != outIndex)
                    values[i] = row.get(i);
                else {
                    Row r = RowFactory.create(res.getPointID(), Arrays.asList(res.getLabels()), Arrays.asList(res.getScores()), Arrays.asList(res.getPositiveThreshold()));
                    values[i] = r;
                }
            }
            return RowFactory.create(values);
        });
        destroyBroadcastVariables();
        return df.sqlContext().createDataFrame(updatedRows, updatedSchema);
    }


    /**
     * Declare all the necessary broadcast variables. The subclasses need to store the broadcast variables somewhere
     * at its internal.
     *
     * @param sc The spark context.
     */
    protected abstract void initBroadcastVariables(JavaSparkContext sc);

    /**
     * Classify the specified point.
     *
     * @param inputPoint The  input point to be classified.
     * @return The classification results.
     */
    protected abstract PointClassificationResults classifyPoint(MultilabelPoint inputPoint);

    /**
     * Destroy the the broadcast variables previously defined with {@link #initBroadcastVariables(JavaSparkContext)}.
     */
    protected abstract void destroyBroadcastVariables();

    @Override
    public StructType transformSchema(StructType structType) {
        DataType inputType = structType.apply(getInputCol()).dataType();
        this.validateInputType(inputType);
        List<String> names = Arrays.asList(structType.fieldNames());
        Cond.require(!names.contains(getOutputCol()), "The output column " + getOutputCol() + " already exists in this schema!");
        List<StructField> fields = new ArrayList<>();
        for (int i = 0; i < structType.fields().length; i++) {
            fields.add(structType.fields()[i]);
        }
        DataType outDataType = DataUtils.pointClassificationResultsDataType();
        fields.add(DataTypes.createStructField(getOutputCol(), outDataType, false));
        return DataTypes.createStructType(fields);
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
