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

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Define logging utility methods.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class Logging {

    /**
     * The name of the logger used inside nlp4sparkml library.
     */
    public static final String NLP_4_SPARK_ML_LOGGER = "nlp4sparkml";

    /**
     * Get the logger used by the library.
     *
     * @return The logger used by the library.
     */
    public static Logger l() {
        return Logger.getLogger(NLP_4_SPARK_ML_LOGGER);
    }

    /**
     * Disable all logging from Spark.
     *
     */
    public static void disableSparkLogging() {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
    }

    /**
     * Disable all logging from the library.
     */
    public static void disableNLP4SparkMLLogging() {
        l().setLevel(Level.OFF);
    }
}
