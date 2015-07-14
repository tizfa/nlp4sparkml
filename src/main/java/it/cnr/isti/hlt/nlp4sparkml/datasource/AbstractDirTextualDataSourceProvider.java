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

package it.cnr.isti.hlt.nlp4sparkml.datasource;

import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public abstract class AbstractDirTextualDataSourceProvider<T extends TextualDocument> extends AbstractTextualDataSourceProvider<T> {

    private String inputDir;

    public AbstractDirTextualDataSourceProvider(String inputDir) {
        Cond.requireNotNull(inputDir, "inputDir");
        Cond.require(!inputDir.isEmpty(), "The input dir is empty");
        this.inputDir = inputDir;
    }

    /**
     * Get the HDFS or local directory containing the input data.
     *
     * @return The HDFS or local directory containing the input data.
     */
    public String getInputDir() {
        return inputDir;
    }

    /**
     * Set the HDFS or local directory containin the input data.
     *
     * @param inputDir The input directory.
     */
    public void setInputDir(String inputDir) {
        Cond.requireNotNull(inputDir, "inputDir");
        Cond.require(!inputDir.isEmpty(), "The input dir is empty");
        this.inputDir = inputDir;
    }
}
