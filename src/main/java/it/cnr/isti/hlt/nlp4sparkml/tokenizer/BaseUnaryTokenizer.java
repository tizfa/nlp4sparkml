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

package it.cnr.isti.hlt.nlp4sparkml.tokenizer;

import it.cnr.isti.hlt.nlp4sparkml.data.DataUtils;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.UnaryTransformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.param.Param;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;

import java.util.Arrays;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class BaseUnaryTokenizer extends UnaryTransformer {

    private final Param<String> tokenPrefix;

    public BaseUnaryTokenizer() {
        tokenPrefix = new Param<String>(this, "tokenPrefix", "The prefix to add to each extracted token");
        setDefault(tokenPrefix, "");
    }

    public Param<String> tokenPrefix() {
        return tokenPrefix;
    }

    /**
     * Get the prefix to add in front of each extracted token.
     *
     * @return The token prefix to add to each extracted token.
     */
    public String getTokenPrefix() {
        return getOrDefault(tokenPrefix);
    }


    public BaseUnaryTokenizer setTokenPrefix(String prefix) {
        set(this.tokenPrefix, prefix);
        return this;
    }

    @Override
    public BaseUnaryTokenizer setInputCol(String inputCol) {
        return (BaseUnaryTokenizer) super.setInputCol(inputCol);
    }

    @Override
    public BaseUnaryTokenizer setOutputCol(String outputCol) {
        return (BaseUnaryTokenizer) super.setOutputCol(outputCol);
    }

    @Override
    public DataType getOutputDataType() {
        ArrayType arType = DataTypes.createArrayType(DataTypes.StringType, false);
        return arType;
    }

    @Override
    protected boolean isOutputDataTypeNullable() {
        return false;
    }


    @Override
    public DataFrame transform(DataFrame dataFrame) {
        String tokenPrefix = getTokenPrefix();
        String inputCol = getInputCol();
        String outputCol = getOutputCol();
        Cond.requireNotNull(dataFrame, "dataFrame");
        StructType updatedSchema = transformSchema(dataFrame.schema());
        JavaRDD<Row> rows = dataFrame.javaRDD();
        JavaRDD<Row> updatedRows = rows.map(row -> {
            int inIndex = row.fieldIndex(inputCol);
            String text = row.getString(inIndex);
            List<String> tokens = extractTokens(tokenPrefix, text);
            Object[] values = new Object[row.size() + 1];
            for (int i = 0; i < row.length(); i++) {
                values[i] = row.get(i);
            }
            values[values.length - 1] = tokens.toArray(new String[0]);

            return RowFactory.create(values);
        });
        return dataFrame.sqlContext().createDataFrame(updatedRows, updatedSchema);
    }


    @Override
    public void validateInputType(DataType inputType) {
        Cond.requireNotNull(inputType, "inputType");
        Cond.require(inputType instanceof StringType, "The type of 'inputType' parameter must be 'StringType'");
    }


    /**
     * Extract the tokens from the specified text.
     *
     * @param text The text to be analyzed.
     * @return The set of tokens extracted. The set can be empty.
     */
    protected abstract List<String> extractTokens(String tokenPrefix, String text);
}
