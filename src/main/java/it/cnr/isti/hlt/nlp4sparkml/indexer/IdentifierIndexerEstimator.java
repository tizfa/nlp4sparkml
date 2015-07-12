/*
 * *****************
 *  Copyright 2015 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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
 * *******************
 */

package it.cnr.isti.hlt.nlp4sparkml.indexer;

import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.UID;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.param.Param;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

/**
 * Created by Tiziano on 11/07/2015.
 */
public class IdentifierIndexerEstimator extends Estimator<IdentifierIndexerModel> {

    private final String uid;
    private final Param<List<String>> featuresFields;


    public IdentifierIndexerEstimator() {
        this.uid = UID.generateUID(getClass());
        featuresFields = new Param<>(this, "featuresFields", "The set of data frame fields containing the features to identify");
        ArrayList<String> feats = new ArrayList<>();
        feats.add("features");
        setDefault(featuresFields, feats);
    }

    public Param<List<String>> featuresFields() {
        return featuresFields;
    }

    public List<String> getFeaturesFields() {
        return getOrDefault(featuresFields);
    }

    public IdentifierIndexerEstimator setFeaturesFields(List<String> featuresFields) {
        Cond.requireNotNull(featuresFields, "featuresFields");
        Cond.require(featuresFields.size() > 0, "The set of features fields is empty");
        set(this.featuresFields, featuresFields);
        return this;
    }

    @Override
    public IdentifierIndexerModel fit(DataFrame dataset) {
        Cond.requireNotNull(dataset, "dataset");
        List<String> fields = getFeaturesFields();
        Cond.require(fields.size() > 0, "The set of input fields must no be empty");
        Column[] cols = new Column[fields.size()];
        for (int i = 0; i < fields.size(); i++) {
            cols[i] = dataset.col(fields.get(i));
        }
        JavaPairRDD<String, Long> identifiers =  dataset.select(cols).toJavaRDD().flatMap(row -> {
            HashMap<String, String> mapItems = new HashMap<>();
            ArrayList<String> ret = new ArrayList<>();
            for (int i = 0; i < row.length(); i++) {
                List<String> items = row.getList(i);
                for (String item : items) {
                    if (!mapItems.containsKey(item)) {
                        mapItems.put(item, item);
                        ret.add(item);
                    }
                }
            }
            return ret;
        }).distinct().zipWithIndex().persist(StorageLevel.MEMORY_AND_DISK_SER());
        return null;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema;
    }

    @Override
    public String uid() {
        return uid;
    }
}
