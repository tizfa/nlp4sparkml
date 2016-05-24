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

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.datasource.LabeledTextualDocument;
import it.cnr.isti.hlt.nlp4sparkml.datasource.SetType;
import it.cnr.isti.hlt.nlp4sparkml.datasource.reuters21578.Reuters21578DataSourceProvider;
import it.cnr.isti.hlt.nlp4sparkml.datasource.reuters21578.Reuters21578SplitType;
import org.apache.commons.lang3.SystemUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class Reuters21578Indexing {

    public static void main(String[] args) {
        // Fix for Windows while using Spark in local mode.
        if (SystemUtils.IS_OS_WINDOWS) {
            System.setProperty("hadoop.home.dir", "f:/winutil/");
        }
        SparkConf conf = new SparkConf();

        // Create Spark context.
        JavaSparkContext sc = new JavaSparkContext("local[*]", "test", conf);

        // Configure the Reuters21578 data source provider.
        String inputDir = "F:\\Utenti\\fagni\\Documenti\\dataset\\reuters21578";
        Reuters21578DataSourceProvider provider = new Reuters21578DataSourceProvider(inputDir);
        provider.setSplitType(Reuters21578SplitType.APTE);
        provider.setDocumentSetType(SetType.TRAINING);

        // Get an RDD of documents from provider.
        JavaRDD<LabeledTextualDocument> docs = provider.readData(sc);

        // Convert the RDD into a dataframe.
        DataFrame df = DataUtils.toLabeledTextualDocumentDataFrame(docs);


    }
}
