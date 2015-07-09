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

import it.cnr.isti.hlt.nlp4sparkml.tokenizer.PuntuactionTokenizer;
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
public class PuntuactionTokenizerTest {

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
        SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);

        ArrayList<DocSample> docs = new ArrayList<>();
        for (int i = 0; i < text.length; i++) {
            docs.add(new DocSample(i, text[i]));
        }
        JavaRDD<DocSample> rdd = sc.parallelize(docs);
        return sqlContext.createDataFrame(rdd, DocSample.class);
    }


    @Test
    public void pipelineTest() {
        SparkConf conf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext("local", "test", conf);
        try {
            DataFrame df = loadInitialData(sc);
            PuntuactionTokenizer tokenizer = new PuntuactionTokenizer();
            tokenizer.setInputCol("content").setOutputCol("tokens");
            DataFrame dfOut = tokenizer.transform(df);
            Row[] rows = dfOut.select(dfOut.col("docID"), dfOut.col("tokens")).take(3);
            for (Row row : rows) {
                System.out.println("Row docID: " + row.getInt(0));
                List<String> tokens = row.getList(1);
                for (String token : tokens) {
                    Assert.assertTrue(!token.isEmpty());
                    Assert.assertTrue(token.equals(token.toLowerCase()));
                    Assert.assertTrue(!token.contains(","));
                    Assert.assertTrue(!token.contains("."));
                    Assert.assertTrue(!token.contains(";"));
                    Assert.assertTrue(!token.contains("<"));
                    Assert.assertTrue(!token.contains("["));
                    Assert.assertTrue(!token.contains("]"));
                    Assert.assertTrue(!token.contains("{"));
                    Assert.assertTrue(!token.contains("}"));
                    Assert.assertTrue(!token.contains("#"));
                    Assert.assertTrue(!token.contains("*"));
                    Assert.assertTrue(!token.contains("?"));
                    Assert.assertTrue(!token.contains("!"));
                    Assert.assertTrue(!token.contains("|"));
                }
            }
        } finally {
            if (sc != null)
                sc.stop();
        }
    }
}
