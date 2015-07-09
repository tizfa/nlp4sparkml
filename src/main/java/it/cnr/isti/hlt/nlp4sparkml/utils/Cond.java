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

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class Cond {
    /**
     * Check if the specified condition is true otherwise it raises an exception
     * specifying the given error message.
     *
     * @param condition
     * @param errorMsg
     */
    public static void require(boolean condition, String errorMsg) {
        if (!condition)
            throw new IllegalArgumentException(errorMsg);
    }

    public static void requireNotNull(Object obj, String paramName) {
        if (obj == null)
            throw new NullPointerException("The object specified by '" + paramName + "' is 'null'");
    }
}
