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

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.indexer.OccurencesCounterHelper;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.param.Param;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;

/**
 * A transformer computing BM25 weights for document features.
 * <p>
 * <br/>
 * <br/>
 * This is an implementation of BM25 based on description in
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.97.7340 .
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BM25WeighterModel extends AbstractStandardWeighter<BM25WeighterModel> {

    private final long numDocuments;
    private final HashMap<Long, Long> featuresDictintDocuments;
    private final double avgDocLength;
    private final Param<Double> k1Param;
    private final Param<Double> bParam;

    public BM25WeighterModel(long numDocuments, Map<Long, Long> featuresDictintDocuments, double avgDocLength) {
        Cond.require(numDocuments > 0, "The number of documents must be greater than 0");
        Cond.requireNotNull(featuresDictintDocuments, "featuresDictintDocuments");
        Cond.require(avgDocLength > 0, "The average document length must be greater than 0");
        this.numDocuments = numDocuments;
        this.featuresDictintDocuments = new HashMap<>();
        this.featuresDictintDocuments.putAll(featuresDictintDocuments);
        this.avgDocLength = avgDocLength;
        k1Param = new Param<Double>(this, "k1Param", "The k1 parameter");
        setDefault(k1Param, 1.2d);
        bParam = new Param<Double>(this, "bParam", "The b parameter");
        setDefault(bParam, 0.5d);
    }


    // ------ Generated param getter to ensure that Scala params() function works well! --------
    public Param<Double> getK1Param() {
        return k1Param;
    }

    public Param<Double> getBParam() {
        return bParam;
    }


    /**
     * Get the k1 parameter value.
     *
     * @return The k1 parameter value.
     */
    public Double getK1() {
        return getOrDefault(k1Param);
    }


    public BM25WeighterModel setK1(double k1) {
        set(this.k1Param, k1);
        return this;
    }

    /**
     * Get the b parameter value.
     *
     * @return The b parameter value.
     */
    public Double getB() {
        return getOrDefault(bParam);
    }


    public BM25WeighterModel setB(double b) {
        set(this.bParam, b);
        return this;
    }

    @Override
    public DataFrame transform(DataFrame dataset) {
        String inputCol = getInputCol();
        StructType newSchema = transformSchema(dataset.schema());
        JavaRDD<Row> ds = dataset.toJavaRDD().persist(StorageLevel.MEMORY_AND_DISK());
        long numDocuments = ds.count();
        Map<Long, Long> featuresDictintDocuments = computeDistinctDocuments(ds, inputCol);
        JavaRDD<Row> computed = ds.map(row -> {
            Object[] fields = DataUtils.copyValuesFromRow(row, 1);
            OccurencesCounterHelper helper = OccurencesCounterHelper.getHelper(row, inputCol);
            List<Long> features = helper.getFeatures();
            List<Long> occurrences = helper.getOccurrences();
            double normalization = 0;
            HashMap<Long, Double> weights = new HashMap<Long, Double>();
            long docLength = 0;
            for (int i = 0; i < features.size(); i++) {
                long numOcc = occurrences.get(i);
                docLength += numOcc;
            }
            for (int i = 0; i < features.size(); i++) {
                long featureID = features.get(i);
                long numOcc = occurrences.get(i);
                long featureNumDistinctDocs = featuresDictintDocuments.get(featureID);
                double score = computeBM25(numOcc, numDocuments, featureNumDistinctDocs, docLength, avgDocLength);
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


    protected final double computeBM25(double tf, double numberOfDocuments,
                                       double documentFrequency, long docLength, double avgDocLength) {

        if (documentFrequency == 0)
            return 0;

        double k1 = getK1();
        double b = getB();
        double rsj = Math.log(numberOfDocuments / documentFrequency);
        double K = k1 * ((1 - b) + (b * docLength / avgDocLength));
        double f = ((k1 + 1) * tf) / (K + tf);

        double weight = f * rsj;
        return weight;
    }

}
