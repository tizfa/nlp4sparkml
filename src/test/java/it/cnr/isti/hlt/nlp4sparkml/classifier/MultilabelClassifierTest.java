/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

import it.cnr.isti.hlt.nlp4sparkml.classifier.MultilabelClassifierEstimator;
import it.cnr.isti.hlt.nlp4sparkml.classifier.MultilabelClassifierModel;
import it.cnr.isti.hlt.nlp4sparkml.classifier.boosting.AdaBoostMHEstimator;
import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.data.MultilabelPoint;
import it.cnr.isti.hlt.nlp4sparkml.data.PointClassificationResults;
import it.cnr.isti.hlt.nlp4sparkml.utils.Logging;
import junit.framework.Assert;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Model;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class MultilabelClassifierTest {


    protected DataFrame loadInitialData(JavaSparkContext sc) {
        SQLContext sqlContext = new SQLContext(sc);

        // Generate the RDD containing the points to learn from.
        int numFeatures = 15;
        ArrayList<MultilabelPoint> pts = new ArrayList<>();
        pts.add(new MultilabelPoint(0, numFeatures, new int[]{1, 3, 5, 7, 9}, new double[]{0.34, 0.1, 0.4, 0.8, 0.83}, new int[]{0}));
        pts.add(new MultilabelPoint(1, numFeatures, new int[]{1, 2, 4, 7, 8}, new double[]{0.24, 0.3, 0.4, 0.71, 0.65}, new int[]{0}));
        pts.add(new MultilabelPoint(2, numFeatures, new int[]{0, 2, 4, 7, 11}, new double[]{0.14, 0.3, 0.4, 0.19, 0.16}, new int[]{0}));
        pts.add(new MultilabelPoint(3, numFeatures, new int[]{10, 11, 12, 13, 14}, new double[]{0.14, 0.3, 0.4, 0.41, 0.19}, new int[]{1}));
        pts.add(new MultilabelPoint(4, numFeatures, new int[]{3, 11, 12, 13, 14}, new double[]{0.1, 0.4, 0.2, 0.51, 0.39}, new int[]{1}));
        pts.add(new MultilabelPoint(5, numFeatures, new int[]{11, 12, 13}, new double[]{0.1, 0.4, 0.2}, new int[]{1}));
        JavaRDD<MultilabelPoint> points = sc.parallelize(pts);
        JavaRDD<Row> rows = points.map(pt->{
            Row ptRow = DataUtils.fromMultilabelPoint(pt);
            return RowFactory.create(ptRow);
        });

        // Create the structure containing this data.
        DataType dt = DataUtils.multilabelPointDataType();
        StructField sf = DataTypes.createStructField("trainingData", dt, false);
        ArrayList fields = new ArrayList<>();
        fields.add(sf);
        StructType schema = DataTypes.createStructType(fields);

        DataFrame df = sqlContext.createDataFrame(rows, schema);
        return df;
    }


    @Test
    public void fitTest() {
        Logging.disableSparkLogging();
        Logging.disableNLP4SparkMLLogging();

        SparkConf conf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext("local", "test", conf);

        try {
            DataFrame df = loadInitialData(sc);
            MultilabelClassifierEstimator estimator = createMultilabelEstimator();
            estimator.setInputCol("trainingData");
            MultilabelClassifierModel model = (MultilabelClassifierModel) estimator.fit(df);
            Assert.assertTrue(model != null);
            Assert.assertTrue(model.parent().uid().equals(estimator.uid()));
            DataFrame dfResults = model.setInputCol("trainingData").setOutputCol("results").transform(df);
            JavaRDD<PointClassificationResults> results = dfResults.select(dfResults.col("results")).javaRDD().map(row->{
                return DataUtils.toPointClassificationResults(row, "results");
            }).cache();
            Assert.assertTrue(results.count() == df.count());

        } finally {
            if (sc != null)
                sc.stop();
        }
    }

    protected abstract MultilabelClassifierEstimator createMultilabelEstimator();
}
