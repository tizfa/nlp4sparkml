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

package it.cnr.isti.hlt.nlp4sparkml.datasource;

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.datasource.reuters21578.Reuters21578DataSourceProvider;
import it.cnr.isti.hlt.nlp4sparkml.datasource.reuters21578.Reuters21578SplitType;
import it.cnr.isti.hlt.nlp4sparkml.indexer.IdentifierGenerator;
import it.cnr.isti.hlt.nlp4sparkml.indexer.IdentifierGeneratorModel;
import it.cnr.isti.hlt.nlp4sparkml.tokenizer.PuntuactionTokenizer;
import it.cnr.isti.hlt.nlp4sparkml.utils.Logging;
import junit.framework.Assert;
import org.apache.commons.lang3.SystemUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.junit.Test;

import java.io.File;
import java.net.URL;
import java.util.Arrays;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class Reuters21578DataSourceProviderTest {


    @Test
    public void pipelineTest() {
        Logging.disableSparkLogging();
        Logging.disableNLP4SparkMLLogging();

        // Fix for Windows while using Spark in local mode.
        if (SystemUtils.IS_OS_WINDOWS) {
            System.setProperty("hadoop.home.dir", "f:/winutil/");
        }
        SparkConf conf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext("local[*]", "test", conf);
        try {
            URL resourceUrl = getClass().
                    getResource("/reuters21578");
            File dsPath = new File(resourceUrl.toURI());
            Reuters21578DataSourceProvider provider = new Reuters21578DataSourceProvider(dsPath.toString());
            provider.setDocumentSetType(SetType.TRAINING);
            provider.setSplitType(Reuters21578SplitType.APTE);
            // Create dataframe from dataset.
            DataFrame df = DataUtils.toLabeledTextualDocumentDataFrame(provider.readData(sc));

            // Extract tokens.
            PuntuactionTokenizer tokenizer = new PuntuactionTokenizer();
            DataFrame df2 = tokenizer.setInputCol("content").setOutputCol("tokens").transform(df).cache();

            // Indexing tokens.
            IdentifierGenerator indexer = new IdentifierGenerator();
            IdentifierGeneratorModel indexedFeatures = indexer.setFeaturesFields(Arrays.asList("tokens")).fit(df2);
            DataFrame df3 = indexedFeatures.setInputCol(Arrays.asList("tokens")).setOutputCol("featuresIndexed").setIdCol("docID").transform(df2);

            Row[] row = df3.where(df3.col("docID").equalTo("0")).collect();
            System.out.println("Rows: " + row.length);

        } catch (Exception e) {
            Assert.fail();
        } finally {
            if (sc != null)
                sc.stop();
        }
    }
}
