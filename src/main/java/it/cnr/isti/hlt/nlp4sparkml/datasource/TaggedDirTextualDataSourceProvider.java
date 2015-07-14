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
public abstract class TaggedDirTextualDataSourceProvider<T extends TextualDocument> extends AbstractDirTextualDataSourceProvider<T> {

    /**
     * The type of documents to extract from data source.
     */
    private SetType documentSetType;

    public TaggedDirTextualDataSourceProvider(String inputDir) {
        super(inputDir);
        documentSetType = SetType.TRAINING;
    }

    /**
     * Get the type of documents extracted from data source.
     *
     * @return The type of documents extracted from data source.
     */
    public SetType getDocumentSetType() {
        return documentSetType;
    }


    /**
     * Set the type of documents to extract from this data source.
     *
     * @param documentType The document type.
     */
    public void setDocumentSetType(SetType documentType) {
        Cond.requireNotNull(documentType, "documentType");
        this.documentSetType = documentType;
    }
}
