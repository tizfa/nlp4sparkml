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

import it.cnr.isti.hlt.nlp4sparkml.indexer.OccurencesCounterHelper;
import it.cnr.isti.hlt.nlp4sparkml.utils.UnaryEstimator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class TfIdfWeighter extends UnaryEstimator<TfIdfWeighterModel> {

    @Override
    public TfIdfWeighterModel fit(DataFrame dataset) {
        String inputCol = getInputCol();
        JavaRDD<Row> ds = dataset.toJavaRDD().persist(StorageLevel.MEMORY_AND_DISK());
        long numDocuments = ds.count();
        Map<Long, Long> featuresDictintDocuments = computeDistinctDocuments(ds, inputCol);
        return new TfIdfWeighterModel(numDocuments, featuresDictintDocuments);
    }

    private Map<Long, Long> computeDistinctDocuments(JavaRDD<Row> ds, String inputCol) {
        return ds.flatMapToPair(row -> {
            List<Long> features = OccurencesCounterHelper.getHelper(row, inputCol).getFeatures();
            HashMap<Long, Long> distinctFeats = new HashMap<Long, Long>();
            for (long featID : features) {
                if (distinctFeats.containsKey(featID))
                    continue;
                distinctFeats.put(featID, featID);
            }
            ArrayList<Tuple2<Long, Long>> ret = new ArrayList<Tuple2<Long, Long>>();
            Iterator<Long> keys = distinctFeats.keySet().iterator();
            while (keys.hasNext()) {
                ret.add(new Tuple2<Long, Long>(keys.next(), 1l));
            }
            return ret;
        }).reduceByKey((v1, v2) -> v1 + v2).collectAsMap();
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema;
    }

}
