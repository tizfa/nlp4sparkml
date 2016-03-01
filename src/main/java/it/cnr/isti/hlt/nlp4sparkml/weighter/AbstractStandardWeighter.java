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
import it.cnr.isti.hlt.nlp4sparkml.utils.UnaryModel;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

/**
 * A skeleton abstract class for a standard weighter.
 * <p>
 * <br/>
 * <br/>
 * The weighter takes as input a dataframe column (specified with {@link #setInputCol(String)}) which must
 * be a struct containing 2 fields. The field
 * 0 must be of an array of types "long" (representing the feature index) and
 * the field 1 must be an array of types "long" (representing the number of occurrences for each specific
 * feature).
 * <br/>
 * The weighter outputs a dataframe column (specified with {@link #setOutputCol(String)}) which is a struct
 * containing two fields. The field
 * 0 is an array of types "long" (representing the feature index) and
 * the field 1 is an array of types "double" (representing the weights for each specific
 * feature).
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class AbstractStandardWeighter<T extends AbstractStandardWeighter<T>> extends UnaryModel<T> {

    public static final String FEATURES = "features";
    public static final String WEIGHTS = "weights";

    @Override
    protected DataType getOutputDataType() {
        return DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(FEATURES, DataTypes.createArrayType(DataTypes.LongType), false),
                DataTypes.createStructField(WEIGHTS, DataTypes.createArrayType(DataTypes.DoubleType), false),
        });
    }

    @Override
    protected boolean isOutputDataTypeNullable() {
        return false;
    }

    @Override
    protected void validateInputType(DataType inputType) {
        OccurencesCounterHelper.validateInputField(inputType);
    }


    @Override
    /**
     * Set the input column. The column must be a struct containing 2
     * fields. The field 0 must be of an array of types "long" (representing the feature index) and
     * the field 1 must be an array of types "long" (representing the number of occurrences of each specific
     * feature).
     */
    public AbstractStandardWeighter setInputCol(String inputCol) {
        return (AbstractStandardWeighter) super.setInputCol(inputCol);
    }
}
