/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

import java.io.Serializable;

/**
 * A datasource textual document.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class TextualDocument implements Serializable {

    /**
     * Used to indicate if a unique DOC_ID has not been unassigned.
     */
    public static final long UNASSIGNED_DOC_ID = -1;

    private long docID;
    private String docName;
    private String content;

    public TextualDocument(long docID, String docName, String content) {
        Cond.requireNotNull(docName, "docName");
        Cond.requireNotNull(content, "content");
        this.docName = docName;
        this.content = content;
        this.docID = docID;
    }

    /**
     * Get the unique ID in the origin data source assigned to this document.
     *
     * @return The unique ID in the origin data source assigned to this document.
     */
    public long getDocID() {
        return docID;
    }

    /**
     * Set the unique ID in the origin data source to assign to this document.
     *
     * @param docID The uniqe document ID.
     */
    public void setDocID(long docID) {
        this.docID = docID;
    }

    /**
     * Get the content of the document.
     *
     * @return The content of teh document.
     */
    public String getContent() {
        return content;
    }

    /**
     * Get the document name.
     *
     * @return The document name.
     */
    public String getDocName() {
        return docName;
    }

    /**
     * Set the document name.
     *
     * @param docName The document name.
     */
    public void setDocName(String docName) {
        this.docName = docName;
    }

    /**
     * Set the raw content of the document.
     *
     * @param content The raw content of the document.
     */
    public void setContent(String content) {
        this.content = content;
    }
}
