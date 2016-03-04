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

package it.cnr.isti.hlt.nlp4sparkml.weighter;

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.indexer.OccurencesCounterHelper;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;

/**
 * A transformer computing TF-IDF weights for document features.
 * <p>
 * <br/>
 * <br/>
 * This implementation uses logarithmically scaled frequency for TF part. See https://en.wikipedia.org/wiki/Tf%E2%80%93idf
 * for more details about TF-IDF method.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class TfIdfWeighterModel extends AbstractStandardWeighter<TfIdfWeighterModel> {

    private final long numDocuments;
    private final HashMap<Long, Long> featuresDictintDocuments;

    public TfIdfWeighterModel(long numDocuments, Map<Long, Long> featuresDictintDocuments) {
        Cond.require(numDocuments > 0, "The number of documents must be greater than 0");
        Cond.requireNotNull(featuresDictintDocuments, "featuresDictintDocuments");
        this.numDocuments = numDocuments;
        this.featuresDictintDocuments = new HashMap<>();
        this.featuresDictintDocuments.putAll(featuresDictintDocuments);
    }


    @Override
    public DataFrame transform(DataFrame dataset) {
        String inputCol = getInputCol();
        StructType newSchema = transformSchema(dataset.schema());
        JavaRDD<Row> ds = dataset.toJavaRDD().persist(StorageLevel.MEMORY_AND_DISK());
        JavaRDD<Row> computed = ds.map(row -> {
            Object[] fields = DataUtils.copyValuesFromRow(row, 1);
            OccurencesCounterHelper helper = OccurencesCounterHelper.getHelper(row, inputCol);
            List<Long> features = helper.getFeatures();
            List<Long> occurrences = helper.getOccurrences();
            double normalization = 0;
            HashMap<Long, Double> weights = new HashMap<Long, Double>();
            for (int i = 0; i < features.size(); i++) {
                long featureID = features.get(i);
                long numOcc = occurrences.get(i);
                long featureNumDistinctDocs = featuresDictintDocuments.get(featureID);
                double score = computeTfIdf(numOcc, numDocuments, featureNumDistinctDocs);
                normalization += (score * score);
                weights.put(featureID, score);
            }
            normalization = Math.sqrt(normalization);
            for (int i = 0; i < features.size(); i++) {
                long featureID = features.get(i);
                double score = weights.get(featureID);
                weights.put(featureID, score / normalization);
            }


            ArrayList<Long> featsOut = new ArrayList<>();
            ArrayList<Double> weightsOut = new ArrayList<Double>();
            Iterator<Long> keys = weights.keySet().iterator();
            while (keys.hasNext()) {
                long feature = keys.next();
                double weight = weights.get(feature);
                featsOut.add(feature);
                weightsOut.add(weight);
            }
            Row r = RowFactory.create(featsOut.toArray(new Long[0]), weightsOut.toArray(new Double[0]));
            fields[fields.length - 1] = r;
            return RowFactory.create(fields);
        });

        return dataset.sqlContext().createDataFrame(computed, newSchema);
    }


    protected double computeTfIdf(long numOccurrencesInsideDoc,
                                  long numTotalDocuments, long featureFrequency) {
        double tfidf = 0;

        double tmp1 = ((double) numTotalDocuments)
                / ((double) (featureFrequency));
        double tmp2 = Math.log(((double) numOccurrencesInsideDoc)) + 1;
        tfidf = tmp2 * Math.log(tmp1);

        return tfidf;
    }

}
