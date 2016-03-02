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
import it.cnr.isti.hlt.nlp4sparkml.utils.JavaEstimator;
import it.cnr.isti.hlt.nlp4sparkml.utils.JavaModel;
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
import org.apache.spark.storage.StorageLevel;

/**
 * A generic multilabel multiclass classifier in the form of a Spark ML estimator.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class MultilabelClassifierEstimator<T extends JavaModel<T>> extends JavaEstimator<T> {

    private final Param<String> inputCol;
    private final String uid;

    public MultilabelClassifierEstimator() {
        uid = UID.generateUID(getClass());
        inputCol = new Param<String>(this, "inputCol", "Input column data name");
        setDefault(inputCol, "points");
    }


    // ------ Generated param getter to ensure that Scala params() function works well! --------
    public Param<String> inputCol() {
        return inputCol;
    }

    /**
     * Get the input column name.
     *
     * @return The input column name.
     */
    public String getInputCol() {
        return getOrDefault(inputCol);
    }

    /**
     * Set the input column name. The input column must be in format as coded in
     * {@link DataUtils#multilabelPointDataType()} method.
     *
     * @param inputCol The input column name.
     * @return This classifier.
     */
    public MultilabelClassifierEstimator<T> setInputCol(String inputCol) {
        set(this.inputCol, inputCol);
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
        //DataFrame df = dataset.withColumn(getOutputCol(), dataset.col(getInputCol()));
        DataFrame df = dataset;
        JavaRDD<Row> rows = df.javaRDD().persist(StorageLevel.MEMORY_AND_DISK_SER());
        int nf = DataUtils.computeNumFeaturesFromDataFrame(rows, getInputCol());
        JavaSparkContext sc = new JavaSparkContext(rows.context());
        Broadcast<Integer> numFeatures = sc.broadcast(nf);
        JavaRDD<MultilabelPoint> inputPoints = rows.map(row -> {
            MultilabelPoint pt = DataUtils.toMultilabelPoint(row, getInputCol(), numFeatures.value());
            return pt;
        });

        initBroadcastVariables(sc);
        T model = buildClassifier(inputPoints, nf);
        model.setParent(this);
        destroyBroadcastVariables();
        return model;
    }

    /**
     * Build the specific learning model by processing input training points available in the specified RDD.
     *
     * @param inputPoints The set of input points used as training data. The input points are not cached.
     * @param numFeatures The total number of features available in training dataset.
     * @return The corresponding classifier.
     */
    protected abstract T buildClassifier(JavaRDD<MultilabelPoint> inputPoints, int numFeatures);


    @Override
    public StructType transformSchema(StructType structType) {
        return structType;
    }


    @Override
    public String uid() {
        return uid;
    }
}
