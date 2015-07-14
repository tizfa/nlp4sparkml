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
 * A textual document with labels assigned to it.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class TextualDocumentWithLabels extends TextualDocument {

    private String[] labels;

    public TextualDocumentWithLabels(long docID, String docName, String content, String[] labels) {
        super(docID, docName, content);
        Cond.requireNotNull(labels, "labels");
        this.labels = labels;
    }

    /**
     * Get the set of labels assigned to this document.
     *
     * @return The set of labels assigned.
     */
    public String[] getLabels() {
        return labels;
    }

    /**
     * Set the labels assigned to this document.
     *
     * @param labels The set of labels assigned to this document.
     */
    public void setLabels(String[] labels) {
        this.labels = labels;
    }
}
