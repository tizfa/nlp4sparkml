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
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BM25Weighter extends UnaryEstimator<BM25WeighterModel> {

    @Override
    public BM25WeighterModel fit(DataFrame dataset) {
        String inputCol = getInputCol();
        JavaRDD<Row> ds = dataset.toJavaRDD().persist(StorageLevel.MEMORY_AND_DISK());
        long numDocuments = ds.count();
        Map<Long, Long> featuresDictintDocuments = computeDistinctDocuments(ds, inputCol);
        double avgLength = computeAvgDocLength(ds, inputCol);
        return new BM25WeighterModel(numDocuments, featuresDictintDocuments, avgLength);
    }

    private Map<Long, Long> computeDistinctDocuments(JavaRDD<Row> ds, String inputCol) {
        return ds.flatMapToPair(row -> {
            OccurencesCounterHelper helper = OccurencesCounterHelper.getHelper(row, inputCol);
            List<Long> features = helper.getFeatures();
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


    private double computeAvgDocLength(JavaRDD<Row> ds, String inputCol) {
        long numDocs = ds.count();
        JavaSparkContext sc = new JavaSparkContext(ds.context());
        Accumulator<Double> accum = sc.accumulator(0d);
        ds.foreach(row -> {
            List<Long> occurrences = OccurencesCounterHelper.getHelper(row, inputCol).getOccurrences();
            long v = occurrences.stream().mapToLong(Long::longValue).sum();
            accum.add((double) v);
        });
        double avgLength = accum.value() / numDocs;
        return avgLength;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema;
    }

}
