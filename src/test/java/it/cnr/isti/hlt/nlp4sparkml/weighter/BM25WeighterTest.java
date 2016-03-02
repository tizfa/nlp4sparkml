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

package it.cnr.isti.hlt.nlp4sparkml.weighter;

import it.cnr.isti.hlt.nlp4sparkml.indexer.IdentifierGenerator;
import it.cnr.isti.hlt.nlp4sparkml.indexer.IdentifierGeneratorModel;
import it.cnr.isti.hlt.nlp4sparkml.indexer.OccurrencesCounter;
import it.cnr.isti.hlt.nlp4sparkml.tokenizer.PuntuactionTokenizer;
import it.cnr.isti.hlt.nlp4sparkml.utils.Logging;
import junit.framework.Assert;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.junit.Test;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BM25WeighterTest {

    private final static String[] text = {
            "This is the first test, yes the first! Uaooo, such a test.",
            "Here the second one and the...",
            "The third one, the most cool."
    };


    public static class DocSample implements Serializable {
        private final long docID;
        private final String content;

        public DocSample(int docID, String content) {
            this.docID = docID;
            this.content = content;
        }

        public long getDocID() {
            return docID;
        }

        public String getContent() {
            return content;
        }
    }


    protected DataFrame loadInitialData(JavaSparkContext sc) {
        SQLContext sqlContext = new SQLContext(sc);

        ArrayList<DocSample> docs = new ArrayList<>();
        for (int i = 0; i < text.length; i++) {
            docs.add(new DocSample(i, text[i]));
        }
        JavaRDD<DocSample> rdd = sc.parallelize(docs);
        return sqlContext.createDataFrame(rdd, DocSample.class);
    }

    @Test
    public void pipelineTest() {
        Logging.disableSparkLogging();
        Logging.disableNLP4SparkMLLogging();
        SparkConf conf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext("local", "test", conf);
        try {
            DataFrame df = loadInitialData(sc);
            PuntuactionTokenizer tokenizer = new PuntuactionTokenizer();
            tokenizer.setInputCol("content").setOutputCol("tokens");
            DataFrame dfFeatures = tokenizer.transform(df);

            IdentifierGenerator identifierEstimator = new IdentifierGenerator();
            ArrayList<String> featuresFields = new ArrayList<>();
            featuresFields.add("tokens");
            identifierEstimator.setFeaturesFields(featuresFields);
            IdentifierGeneratorModel identifierIndexer = identifierEstimator.fit(dfFeatures);
            DataFrame dfMapping = identifierIndexer.getInternalFeaturesMappinng().cache();
            Row[] features = dfMapping.collect();
            System.out.println("---- FEATURES----");
            for (int i = 0; i < features.length; i++) {
                System.out.println(features[i].toString());
            }

            System.out.println("---- DATASET INDEXED----");
            Assert.assertTrue(identifierIndexer != null);
            DataFrame dfIndexedFeatures = identifierIndexer.setIdCol("docID").setInputCol(featuresFields).setOutputCol("featuresIndexed").transform(dfFeatures).cache();
            Assert.assertTrue(dfIndexedFeatures.count() == 3);
            Row[] rows = dfIndexedFeatures.take(3);
            for (Row r : rows) {
                System.out.println(r.toString());
            }

            System.out.println("---- DATASET INDEXED WITH OCCURRENCES----");
            OccurrencesCounter counter = new OccurrencesCounter();
            counter.setInputCol("featuresIndexed").setOutputCol("occurrences");
            DataFrame dfOccur = counter.transform(dfIndexedFeatures);
            List<Row> r = dfOccur.select(dfOccur.col("featuresIndexed"), dfOccur.col("occurrences")).collectAsList();
            for (Row ro : r)
                System.out.println(ro);

            BM25Weighter weighter = new BM25Weighter();
            BM25WeighterModel bm25Model = weighter.setInputCol("occurrences").fit(dfOccur);
            DataFrame dfWeights = bm25Model.setInputCol("occurrences").setOutputCol("weights").transform(dfOccur);
            r = dfWeights.select(dfWeights.col("weights")).collectAsList();
            for (Row ro : r)
                System.out.println(ro);

        } finally {
            if (sc != null)
                sc.stop();
        }
    }
}
