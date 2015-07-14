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

package it.cnr.isti.hlt.nlp4sparkml.datasource;

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.datasource.reuters21578.Reuters21578DataSourceProvider;
import it.cnr.isti.hlt.nlp4sparkml.datasource.reuters21578.Reuters21578SplitType;
import it.cnr.isti.hlt.nlp4sparkml.indexer.IdentifierIndexer;
import it.cnr.isti.hlt.nlp4sparkml.indexer.IdentifierIndexerModel;
import it.cnr.isti.hlt.nlp4sparkml.tokenizer.PuntuactionTokenizer;
import it.cnr.isti.hlt.nlp4sparkml.utils.Logging;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.junit.Test;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class Reuters21578DataSourceProviderTest {

    private final static String[] text = {
            "This is the first test! Uaooo!",
            "Here the second one and...",
            "The third one, the most cool."
    };


    public static class DocSample implements Serializable {
        private final int docID;
        private final String content;

        public DocSample(int docID, String content) {
            this.docID = docID;
            this.content = content;
        }

        public int getDocID() {
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
        System.setProperty("hadoop.home.dir", "F:/winutil/");
        SparkConf conf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext("local[8]", "test", conf);
        try {
            Reuters21578DataSourceProvider provider = new Reuters21578DataSourceProvider("F:/Utenti/fagni/Documenti/dataset/reuters21578");
            provider.setDocumentSetType(SetType.TRAINING);
            provider.setSplitType(Reuters21578SplitType.APTE);
            // Create dataframe from dataset.
            DataFrame df = DataUtils.toTextualDocumentWithlabelsDataFrame(provider.readData(sc));

            // Extract tokens.
            PuntuactionTokenizer tokenizer = new PuntuactionTokenizer();
            df = tokenizer.setInputCol("content").setOutputCol("tokens").transform(df).cache();

            // Indexing tokens.
            IdentifierIndexer indexer = new IdentifierIndexer();
            IdentifierIndexerModel indexedFeatures = indexer.setFeaturesFields(Arrays.asList("tokens")).fit(df);
            df = indexedFeatures.setInputCol(Arrays.asList("tokens")).setOutputCol("features").setIdCol("docID").transform(df);

            Row[] row = df.collect();
            System.out.println("Rows: " + row.length);

        } finally {
            if (sc != null)
                sc.stop();
        }
    }
}