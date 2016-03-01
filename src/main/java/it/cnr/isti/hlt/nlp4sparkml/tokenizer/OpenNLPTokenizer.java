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

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.StructType;

/**
 * A tokenizer implementation based on OpenNLP tokenizer.<br/><br/>
 * See <a href="https://opennlp.apache.org/documentation/1.5.3/manual/opennlp.html#tools.tokenizer">OpenNLP tokenizer</a> for
 * more details about its features.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class OpenNLPTokenizer extends Transformer {
    @Override
    public DataFrame transform(DataFrame dataFrame) {
        return null;
    }

    @Override
    public OpenNLPTokenizer copy(ParamMap extra) {
        return null;
    }

    @Override
    public StructType transformSchema(StructType structType) {
        return null;
    }

    @Override
    public String uid() {
        return null;
    }
}
