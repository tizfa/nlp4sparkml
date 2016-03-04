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

package it.cnr.isti.hlt.nlp4sparkml.indexer;

import it.cnr.isti.hlt.nlp4sparkml.utils.BaseHelper;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

/**
 * This is an helper class useful to work with features occurrences.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class OccurencesCounterHelper extends BaseHelper {


    private OccurencesCounterHelper(Row row, String fieldName) {
        super(row, fieldName);
    }


    /**
     * Create a field value containing two fields. The field 0 is an array of long containing the set of
     * features IDs considered, the the field 1 is an array of long containing the corresponding number of
     * occurrences for each feature.
     *
     * @param counter The map containing the association between a feature ID and its number of occurrences.
     * @return A complex field filled with features and occurrences.
     */
    public static Row createOccurrencesField(HashMap<Long, Long> counter) {
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
        return r;
    }


    /**
     * Validate the specified input type to be compatible with type used to
     * represent occurrences.
     *
     * @param inputType The input type to be validated.
     */
    public static void validateInputField(DataType inputType) {
        Cond.require(inputType instanceof StructType, "The input type must be an instance of struct type");
        StructType st = (StructType) inputType;
        Cond.require(st.fields().length == 2, "The number of fields must be 2");
        Cond.require(st.fields()[0].dataType() instanceof ArrayType, "The field 0 must have type ArrayType");
        Cond.require(st.fields()[1].dataType() instanceof ArrayType, "The field 0 must have type ArrayType");
    }


    public static OccurencesCounterHelper getHelper(Row row, String column) {
        Cond.requireNotNull(row, "row");
        Cond.requireNotNull(column, "column");
        Row occurrencesStruct = row.getStruct(row.fieldIndex(column));
        try {
            validateInputField(occurrencesStruct.schema());
        } catch (Exception e) {
            throw new RuntimeException("Validating field schema", e);
        }
        return new OccurencesCounterHelper(row, column);
    }

    /**
     * Get the set of features IDs stored in the specified column of the given row.
     *
     * @return The set of features IDs.
     */
    public List<Long> getFeatures() {
        Row occurrencesStruct = getRow().getStruct(getRow().fieldIndex(getFieldName()));
        return occurrencesStruct.getList(0);
    }

    /**
     * Get the set of occurrences for the features stored in the specified
     * column of the given row. The set of occurrences corresponds to the set of
     * features available with {@link #getFeatures()} method.
     *
     * @return The set of features occurrences.
     */
    public List<Long> getOccurrences() {
        Row occurrencesStruct = getRow().getStruct(getRow().fieldIndex(getFieldName()));
        return occurrencesStruct.getList(1);
    }
}
