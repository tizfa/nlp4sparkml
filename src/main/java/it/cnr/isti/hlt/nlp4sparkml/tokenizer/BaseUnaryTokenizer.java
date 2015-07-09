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

import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.UnaryTransformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;

import java.util.Arrays;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class BaseUnaryTokenizer extends UnaryTransformer {

    public BaseUnaryTokenizer() {
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
        Cond.requireNotNull(dataFrame, "dataFrame");
        StructType updatedSchema = transformSchema(dataFrame.schema());
        DataFrame df = dataFrame.withColumn(getOutputCol(), dataFrame.col(getInputCol()));
        JavaRDD<Row> rows = df.javaRDD();
        JavaRDD<Row> updatedRows = rows.map(row -> {
            int inIndex = row.fieldIndex(getInputCol());
            int outIndex = row.fieldIndex(getOutputCol());
            String text = row.getString(inIndex);
            List<String> tokens = extractTokens(text);
            Object[] values = new Object[row.size()];
            for (int i = 0; i < row.size(); i++) {
                if (i != outIndex)
                    values[i] = row.get(i);
                else
                    values[i] = tokens;
            }
            return RowFactory.create(values);
        });
        return df.sqlContext().createDataFrame(updatedRows, updatedSchema);
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
    protected abstract List<String> extractTokens(String text);
}
