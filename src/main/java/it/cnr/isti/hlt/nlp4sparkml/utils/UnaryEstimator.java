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

package it.cnr.isti.hlt.nlp4sparkml.utils;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.Param;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class UnaryEstimator<T extends Model<T>> extends Estimator<T> {
    private final Param<String> inputColParam;
    private final String uid;

    public UnaryEstimator() {
        uid = UID.generateUID(getClass());
        inputColParam = new Param<String>(this, "inputCol", "Input column name");
        setDefault(inputColParam, "inputCol");
    }


    // ------ Generated param getter to ensure that Scala params() function works well! --------
    public Param<String> getInputColParam() {
        return inputColParam;
    }


    /**
     * Get the input column name.
     *
     * @return The input column name.
     */
    public String getInputCol() {
        return getOrDefault(inputColParam);
    }


    public UnaryEstimator<T> setInputCol(String inputCol) {
        set(this.inputColParam, inputCol);
        return this;
    }


    @Override
    public String uid() {
        return uid;
    }
}
