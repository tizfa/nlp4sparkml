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

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;

import java.util.Iterator;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class StageHelper {

    /**
     * Create a new instance of the source transformer, copying all
     * its parameters to the new instance.
     *
     * @param src   The source transformer.
     * @param extra The parameters map.
     * @return
     */
    public static Transformer copy(Transformer src, ParamMap extra) {
        try {
            Transformer t = src.getClass().newInstance();
            for (ParamPair p : extra) {

            }
            for (int i = 0; i < extra.size(); i++) {
                extra.toSeq();
                Object val = src.getOrDefault(p);
                t.set(p, val);
            }
            return t;
        } catch (Exception e) {
            throw new RuntimeException("Can not instantiate new instance of " + src.getClass().getName(), e);
        }
    }
}
