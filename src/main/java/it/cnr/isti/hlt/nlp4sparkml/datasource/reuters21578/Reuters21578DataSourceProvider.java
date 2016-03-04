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

package it.cnr.isti.hlt.nlp4sparkml.datasource.reuters21578;

import it.cnr.isti.hlt.nlp4sparkml.datasource.*;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import it.cnr.isti.hlt.nlp4sparkml.utils.Pair;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

/**
 * A Reuters21578 data source provider. The dataset is available here:
 * http://www.daviddlewis.com/resources/testcollections/reuters21578/
 * <br/><br/>
 * NOTE: this class is a porting of an existing code we have used internally at our lab in several contexts,
 * so it is not clean and it would require to be optimized and rewritten better.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class Reuters21578DataSourceProvider extends TaggedDirTextualDataSourceProvider<LabeledTextualDocument> {

    private Reuters21578SplitType splitType;


    public Reuters21578DataSourceProvider(String inputDir) {
        super(inputDir);
        splitType = Reuters21578SplitType.APTE;
    }

    @Override
    public JavaRDD<LabeledTextualDocument> readData(JavaSparkContext sc) {
        Cond.requireNotNull(sc, "sc");
        return sc.wholeTextFiles(getInputDir()).filter(f -> {
            File fi = new File(f._1());
            String fname = fi.getName();
            return fname.startsWith("reut2-") && fname.endsWith(".sgm");
        }).flatMap(f -> {
            String fileContent = f._2();
            ArrayList<LabeledTextualDocument> docs = readDocuments(fileContent);
            return docs;
        }).zipWithIndex().map(v -> {
            v._1().setDocID(v._2());
            return v._1();
        });
    }

    /**
     * Get the split type used while reading the documents.
     *
     * @return The split type used while reading the documents.
     */
    public Reuters21578SplitType getSplitType() {
        return splitType;
    }

    /**
     * Set the split type to use when reading the documents from original
     * Reuters21578 corpus set.
     *
     * @param splitType The type of split to use.
     */
    public void setSplitType(Reuters21578SplitType splitType) {
        Cond.requireNotNull(splitType, "splitType");
        this.splitType = splitType;
    }

    protected ArrayList<LabeledTextualDocument> readDocuments(String fileContent) {
        BufferedReader reader = new BufferedReader(new StringReader(fileContent));
        try {
            try {
                ArrayList<LabeledTextualDocument> docs = new ArrayList<>();
                String[] rawDocLines = readRawDocument(reader);
                while (rawDocLines.length != 0) {
                    LabeledTextualDocument doc = processRawDocument(rawDocLines);
                    if (doc != null) {
                        docs.add(doc);
                    }

                    // Read next document.
                    rawDocLines = readRawDocument(reader);
                }
                return docs;
            } finally {
                if (reader != null)
                    reader.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Reading content of a file", e);
        }
    }

    /**
     * Read a raw document from the current Reuters file.
     *
     * @return The set of lines containing the document.
     */
    protected String[] readRawDocument(BufferedReader reader) {
        boolean docRead = false;
        ArrayList<String> lines = new ArrayList<String>();

        while (!docRead) {
            String line;
            try {
                line = reader.readLine();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            if (line == null) {
                // Reach the end of file.
                String[] linesToReturn = new String[0];
                return linesToReturn;
            }

            if (line.equals("<!DOCTYPE lewis SYSTEM \"lewis.dtd\">")) {
                continue;
            }

            if (line.equals(""))
                continue;

            // Add the line.
            lines.add(line);

            // Check if we have reach the end of a document.
            if (line.equals("</REUTERS>"))
                docRead = true;
        }

        String[] linesToReturn = new String[lines.size()];
        for (int i = 0; i < lines.size(); i++) {
            linesToReturn[i] = lines.get(i);
        }

        return linesToReturn;
    }

    /**
     * Get an high level representation of the passed raw document.
     *
     * @param lines The set of lines representing the raw document to process.
     * @return An high level representation of the document or "null" if current
     * document must be excluded from set of available documents.
     */
    protected LabeledTextualDocument processRawDocument(String[] lines) {
        // Get the discriminating attributes of first line (tag REUTERS)
        String lewisSplit = getAttribute(lines[0], "LEWISSPLIT");
        String topics = getAttribute(lines[0], "TOPICS");
        String cgiSplit = getAttribute(lines[0], "CGISPLIT");

        DocumentType docType = DocumentType.TRAINING;
        Pair<Boolean, DocumentType> res = excludeDocument(lewisSplit, topics,
                cgiSplit, docType);
        if (res.getV1())
            // The document is not contained in the set of currently
            // requested documents tipology.
            return null;

        docType = res.getV2();

        // Get the ID of the document.
        String docName = new String(getAttribute(lines[0], "NEWID"));

        ArrayList<String> categories = new ArrayList<String>();
        String content = "";

        // Analyze all the lines of this document.
        DocIterator it = new DocIterator();
        it.row = 1;
        it.col = 0;
        while (it.row < lines.length - 1) {
            if (lines[it.row].startsWith("<DATE>")) {
                // Analyze the date of the document. SINGLE LINE.
                processDate(lines, it);
            } else if (lines[it.row].startsWith("<MKNOTE>")) {
                // Analyze the note contained in the document. MULTI LINE.
                processNotes(lines, it);
            } else if ((lines[it.row].startsWith("<TOPICS>"))
                    || (lines[it.row].startsWith("<PLACES>"))
                    || (lines[it.row].startsWith("<PEOPLE>"))
                    || (lines[it.row].startsWith("<ORGS>"))
                    || (lines[it.row].startsWith("<EXCHANGES>"))
                    || (lines[it.row].startsWith("<COMPANIES>"))) {
                // Analyze the TOPICS categories, if any, contained in the
                // document. SINGLE LINE.

                String[] cats = processCategories(lines, it);
                for (int i = 0; i < cats.length; ++i)
                    categories.add(new String(cats[i]));
            } else if (lines[it.row].startsWith("<UNKNOWN>")) {
                // Unknown characters sequence contained in the document. MULTI
                // LINE.
                processUnknowns(lines, it);
            } else if (lines[it.row].startsWith("<TEXT")) {
                // Textual material contained in this document. MULTI LINE.
                content += " . \n" + processText(lines, it);
            }
        }


        assert (lines[lines.length - 1].startsWith("</REUTERS>"));

        return new LabeledTextualDocument(TextualDocument.UNASSIGNED_DOC_ID, docName, content, categories.toArray(new String[0]));
    }

    protected String processAuthor(String[] lines, DocIterator it) {
        String author = "";
        String line = lines[it.row];
        it.col = line.indexOf("<AUTHOR>");
        assert (it.col != -1);
        it.col += "<AUTHOR>".length();

        boolean analyzed = false;
        while (!analyzed) {
            int lastPos = line.length();

            if (line.indexOf("</AUTHOR>", it.col) != -1) {
                // Ok. This is the last line.
                lastPos = line.indexOf("</AUTHOR>", it.col);
                analyzed = true;
            }

            // Split the text.
            String lineToSplit = line.substring(it.col, lastPos);
            it.col = lastPos;

            // Get the features contained in this line.
            author += " . /" + lineToSplit;

            if (analyzed)
                it.col += "</AUTHOR>".length();

            if (it.col == line.length()) {
                // Update row and column.
                it.row++;
                it.col = 0;

                if (!analyzed)
                    line = lines[it.row];
            }
        }

        return author;
    }

    protected String processTitle(String[] lines, DocIterator it) {
        String title = "";
        String line = lines[it.row];
        it.col = line.indexOf("<TITLE>");
        assert (it.col != -1);
        it.col += "<TITLE>".length();

        boolean analyzed = false;
        while (!analyzed) {
            int lastPos = line.length();

            if (line.indexOf("</TITLE>", it.col) != -1) {
                // Ok. This is the last line.
                lastPos = line.indexOf("</TITLE>", it.col);
                analyzed = true;
            }

            // Split the text.
            String lineToSplit = line.substring(it.col, lastPos);
            it.col = lastPos;

            // Extract the features from current line.
            title += " . \n" + lineToSplit;

            if (analyzed)
                it.col += "</TITLE>".length();

            if (it.col == line.length()) {
                // Update row and column.
                it.row++;
                it.col = 0;

                if (!analyzed)
                    line = lines[it.row];
            }
        }

        return title;
    }

    protected String processBody(String[] lines, DocIterator it) {
        String body = "";
        String line = lines[it.row];
        it.col = line.indexOf("<BODY>");
        assert (it.col != -1);
        it.col += "<BODY>".length();

        boolean analyzed = false;
        while (!analyzed) {
            int lastPos = line.length();

            if (line.indexOf("</BODY>", it.col) != -1) {
                // Ok. This is the last line.
                lastPos = line.indexOf("</BODY>", it.col);
                analyzed = true;
            }

            // Split the text.
            String lineToSplit = line.substring(it.col, lastPos);
            it.col = lastPos;

            // Extract the features from current line.
            body += " . \n" + lineToSplit;

            if (analyzed)
                it.col += "</BODY>".length();

            if (it.col == line.length()) {
                // Update row and column.
                it.row++;
                it.col = 0;

                if (!analyzed)
                    line = lines[it.row];
            }
        }

        return body;
    }

    protected String processText(String[] lines, DocIterator it) {
        String content = "";

        it.col = 0;

        assert (lines[it.row].indexOf("<TEXT") != -1);

        it.row++;

        String line = lines[it.row];

        boolean processAll = false;
        it.col = 0;
        while (!processAll) {
            if (line.indexOf("<AUTHOR>", it.col) != -1) {
                // Skip this data.
                processAuthor(lines, it);

            } else if (line.indexOf("<DATELINE>", it.col) != -1) {
                // Skip this data.
                processDateline(lines, it);
            } else if (line.indexOf("<TITLE>", it.col) != -1) {
                // Get the title features.
                content += " . \n" + processTitle(lines, it);
            } else if (line.indexOf("<BODY>", it.col) != -1) {
                // Extract the features from data.
                content += " . \n" + processBody(lines, it);

                // Skip irrelevant lines...
                it.col = 0;
                while (!processAll) {
                    if (lines[it.row].indexOf("</TEXT>") != -1)
                        processAll = true;
                    it.row++;
                }
            } else {
                // The article is of type "UNPROC".
                content += " . \n" + processUnproc(lines, it);

                // Skip irrelevant lines...
                it.col = 0;
                processAll = true;
            }

            // Update the line.
            line = lines[it.row];
        }

        return content;
    }

    protected String processUnproc(String[] lines, DocIterator it) {
        String unproc = "";
        String line = lines[it.row];
        it.col = 0;

        boolean analyzed = false;
        while (!analyzed) {
            int lastPos = line.length();

            if (line.indexOf("</TEXT>", it.col) != -1) {
                // Ok. This is the last line.
                lastPos = line.indexOf("</TEXT>", it.col);
                analyzed = true;
            }

            // Split the text.
            String lineToSplit = line.substring(it.col, lastPos);
            it.col = lastPos;

            // Extract the features from current line.
            unproc += " . \n" + lineToSplit;

            if (analyzed)
                it.col += "</TEXT>".length();

            if (it.col == line.length()) {
                // Update row and column.
                it.row++;
                it.col = 0;

                if (!analyzed)
                    line = lines[it.row];
            }
        }

        return unproc;
    }

    protected void processDateline(String[] lines, DocIterator it) {
        String line = lines[it.row];
        it.col = line.indexOf("<DATELINE>");
        assert (it.col != -1);
        it.col += "<DATELINE>".length();

        boolean analyzed = false;
        while (!analyzed) {
            int lastPos = line.length();

            if (line.indexOf("</DATELINE>", it.col) != -1) {
                // Ok. This is the last line.
                lastPos = line.indexOf("</DATELINE>", it.col);
                analyzed = true;
            }

            it.col = lastPos;
            if (analyzed)
                it.col += "</DATELINE>".length();

            if (it.col == line.length()) {
                // Update row and column.
                it.row++;
                it.col = 0;

                if (!analyzed)
                    line = lines[it.row];
            }
        }
    }

    protected void processUnknowns(String[] lines, DocIterator indexRow) {
        // Skip unknowns lines
        boolean processAll = false;
        while (!processAll) {
            String line = lines[indexRow.row];
            if (line.indexOf("</UNKNOWN>") != -1)
                processAll = true;
            indexRow.row++;
        }
    }

    protected String[] processCategories(String[] lines, DocIterator indexRow) {

        String line = lines[indexRow.row];

        // Get the name of parent category.
        int startPos = line.indexOf("<");
        int endPos = line.indexOf(">");
        String nameParent = line.substring(startPos + 1, endPos);

        if ((nameParent.equals("EXCHANGES")) || (nameParent.equals("ORGS"))
                || (nameParent.equals("PEOPLE"))
                || (nameParent.equals("PLACES"))
                || (nameParent.equals("COMPANIES"))) {
            // I need to skip this category...
            // Update the row
            indexRow.row++;
            return new String[]{};
        }

        ArrayList<String> cats = new ArrayList<String>();
        String bPattern = "<D>";
        String ePattern = "</D>";
        int index = endPos + 1;

        while (index < line.length()) {
            int pos = line.indexOf(bPattern, index);
            if (pos == -1) {
                // Ok. Reach the end of categories.
                index = line.length();
                continue;
            } else {
                // Get a category.
                int last = line.indexOf(ePattern, index);
                assert (last != -1);
                String category = line.substring(pos + bPattern.length(), last);

                // Update position
                index = last + ePattern.length();

                // Save category.
                cats.add(category);
            }
        }
        // Update the row
        indexRow.row++;

        return cats.toArray(new String[cats.size()]);
    }

    protected Date processDate(String[] lines, DocIterator indexRow) {
        String start = "<DATE>";
        String end = "</DATE>";

        int startIdx = lines[indexRow.row].indexOf(start) + start.length();
        int endIdx = lines[indexRow.row].indexOf(end);

        String content = lines[indexRow.row].substring(startIdx, endIdx);

        indexRow.row++;
        SimpleDateFormat dateFormat = new SimpleDateFormat();
        try {
            return dateFormat.parse(content);
        } catch (ParseException e) {
        }
        return null;
    }

    protected void processNotes(String[] lines, DocIterator it) {
        String line = lines[it.row];
        it.col = line.indexOf("<MKNOTE>");
        assert (it.col != -1);
        it.col += "<MKNOTE>".length();

        boolean analyzed = false;
        while (!analyzed) {
            int lastPos = line.length();

            if (line.indexOf("</MKNOTE>", it.col) != -1) {
                // Ok. This is the last line.
                lastPos = line.indexOf("</MKNOTE>", it.col);
                analyzed = true;
            }

            it.col = lastPos;

            if (analyzed)
                it.col += "</MKNOTE>".length();

            if (it.col == line.length()) {
                // Update row and column.
                it.row++;
                it.col = 0;

                if (!analyzed)
                    line = lines[it.row];
            }
        }
    }

    /**
     * Get the value of an attribute from the specified input string. The
     * attribute must have the syntax ATTRIBUTE_NAME="VALUE" where
     * ATTRIBUTE_NAME is the name of attribute (parameter "attributeName") and
     * VALUE is the value that must be returned to caller.
     *
     * @param line          The string where to search the specified attribute.
     * @param attributeName The attribute name to search.
     * @return The value of specified attribute.
     */
    protected String getAttribute(String line, String attributeName) {
        String toSearch = attributeName + "=\"";

        // Find the start position.
        int pos = line.indexOf(toSearch);
        assert (pos != -1);
        int startPos = pos + toSearch.length();

        // Find the end position.
        int endPos = line.indexOf("\"", startPos);

        String attr = new String(line.substring(startPos, endPos));
        return attr;
    }

    protected Pair<Boolean, DocumentType> excludeDocument(String lewisSplit,
                                                          String topics, String cgiSplit, DocumentType docType) {
        if (getSplitType() == Reuters21578SplitType.LEWIS) {
            // Lewis split.
            return excludeLewisDocument(lewisSplit, topics, cgiSplit, docType);
        } else if (getSplitType() == Reuters21578SplitType.HAYES) {
            // Hayes split.
            return excludeHayesDocument(lewisSplit, topics, cgiSplit, docType);
        } else {
            // Apte split.
            return excludeApteDocument(lewisSplit, topics, cgiSplit, docType);
        }
    }

    protected Pair<Boolean, DocumentType> excludeLewisDocument(
            String lewisSplit, String topics, String cgiSplit,
            DocumentType docType) {

        if (getDocumentSetType() == SetType.TRAINING) {
            // I want training documents...
            if ((lewisSplit.equals("TRAIN")) && (!topics.equals("BYPASS"))) {
                docType = DocumentType.TRAINING;

                // Valid document.

                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        false, docType);
                return res;
            } else {
                // Invalid document.
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        true, docType);
                return res;
            }
        } else if (getDocumentSetType() == SetType.TEST) {
            // I want test documents...

            if ((lewisSplit.equals("TEST")) && (!topics.equals("BYPASS"))) {
                // Valid document.
                docType = DocumentType.TEST;
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        false, docType);
                return res;
            } else {
                // Invalid document.
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        true, docType);
                return res;
            }
        } else if (getDocumentSetType() == SetType.ALL) {
            if ((lewisSplit.equals("TRAIN")) && (!topics.equals("BYPASS")))
                docType = DocumentType.TRAINING;
            else if ((lewisSplit.equals("TEST")) && (!topics.equals("BYPASS"))) {
                docType = DocumentType.TEST;
            } else
                docType = DocumentType.VALIDATION;

            // I want all documemt types.
            Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                    false, docType);
            return res;
        } else {
            // Skip the document.
            Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                    true, docType);
            return res;
        }
    }

    protected Pair<Boolean, DocumentType> excludeHayesDocument(
            String lewisSplit, String topics, String cgiSplit,
            DocumentType docType) {
        if (getDocumentSetType() == SetType.TRAINING) {
            // I want training documents...
            if (cgiSplit.equals("TRAINING-SET")) {
                // Valid document.
                docType = DocumentType.TRAINING;
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        false, docType);
                return res;
            } else {
                // Invalid document.
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        true, docType);
                return res;
            }
        } else if (getDocumentSetType() == SetType.TEST) {
            // I want test documents...

            if (cgiSplit.equals("PUBLISHED-TESTSET")) {
                // Valid document.
                docType = DocumentType.TEST;
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        false, docType);
                return res;
            } else {
                // Invalid document.
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        true, docType);
                return res;
            }
        } else if (getDocumentSetType() == SetType.ALL) {
            if (cgiSplit.equals("TRAINING-SET")) {
                // Valid document.
                docType = DocumentType.TRAINING;
            } else if (cgiSplit.equals("PUBLISHED-TESTSET"))
                docType = DocumentType.TEST;
            else
                docType = DocumentType.VALIDATION;

            // I want all documemt types.
            Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                    false, docType);
            return res;
        } else {
            Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                    true, docType);
            return res;
        }
    }

    protected Pair<Boolean, DocumentType> excludeApteDocument(
            String lewisSplit, String topics, String cgiSplit,
            DocumentType docType) {
        if (getDocumentSetType() == SetType.TRAINING) {
            // I want training documents...
            if ((lewisSplit.equals("TRAIN")) && (topics.equals("YES"))) {
                // Valid document.
                docType = DocumentType.TRAINING;
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        false, docType);
                return res;
            } else {
                // Invalid document.
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        true, docType);
                return res;
            }
        } else if (getDocumentSetType() == SetType.TEST) {
            // I want test documents...

            if ((lewisSplit.equals("TEST")) && (topics.equals("YES"))) {
                // Valid document.
                docType = DocumentType.TEST;
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        false, docType);
                return res;
            } else {
                // Invalid document.
                Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                        true, docType);
                return res;
            }
        } else if (getDocumentSetType() == SetType.ALL) {
            if ((lewisSplit.equals("TRAIN")) && (topics.equals("YES"))) {
                // Valid document.
                docType = DocumentType.TRAINING;
            } else if ((lewisSplit.equals("TEST")) && (topics.equals("YES")))
                docType = DocumentType.TEST;
            else
                docType = DocumentType.VALIDATION;

            // I want all documemt types.
            Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                    false, docType);
            return res;
        } else {
            Pair<Boolean, DocumentType> res = new Pair<Boolean, DocumentType>(
                    true, docType);
            return res;
        }
    }

}
