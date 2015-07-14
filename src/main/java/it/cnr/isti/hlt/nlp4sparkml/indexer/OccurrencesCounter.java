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

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.UnaryTransformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class OccurrencesCounter extends UnaryTransformer {

    public static final String FEATURES = "features";
    public static final String OCCURRENCES = "occurrences";


    @Override
    protected DataType getOutputDataType() {
        return DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(FEATURES, DataTypes.createArrayType(DataTypes.LongType), false),
                DataTypes.createStructField(OCCURRENCES, DataTypes.createArrayType(DataTypes.LongType), false),
        });
    }

    @Override
    protected boolean isOutputDataTypeNullable() {
        return false;
    }

    @Override
    protected void validateInputType(DataType inputType) {
        Cond.require(inputType instanceof ArrayType, "The input type must be an array type of longs");
    }

    @Override
    public DataFrame transform(DataFrame dataset) {
        String inputCol = getInputCol();
        StructType newSchema = transformSchema(dataset.schema());
        JavaRDD<Row> computed = dataset.toJavaRDD().map(row -> {
            Object[] fields = DataUtils.copyValuesFromRow(row, 1);
            List<Long> features = row.getList(row.fieldIndex(inputCol));
            HashMap<Long, Long> counter = new HashMap<Long, Long>();
            for (long feature : features) {
                if (counter.containsKey(feature))
                    counter.put(feature, counter.get(feature) + 1);
                else
                    counter.put(feature, 1l);
            }
            ArrayList<Long> featsOut = new ArrayList<>();
            ArrayList<Long> occurrencesOut = new ArrayList<Long>();
            Iterator<Long> keys = counter.keySet().iterator();
            while (keys.hasNext()) {
                long feature = keys.next();
                long numOccurrences = counter.get(feature);
                featsOut.add(feature);
                occurrencesOut.add(numOccurrences);
            }
            Row r = RowFactory.create(featsOut.toArray(new Long[0]), occurrencesOut.toArray(new Long[0]));
            fields[fields.length - 1] = r;
            return RowFactory.create(fields);
        });

        return dataset.sqlContext().createDataFrame(computed, newSchema);
    }
}
