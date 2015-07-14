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

package it.cnr.isti.hlt.nlp4sparkml.data;


import it.cnr.isti.hlt.nlp4sparkml.datasource.TextualDocument;
import it.cnr.isti.hlt.nlp4sparkml.datasource.TextualDocumentWithLabels;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class DataUtils {

    public static final String POINT_ID = "pointID";
    public static final String FEATURES = "features";
    public static final String WEIGHTS = "weights";
    public static final String LABELS = "labels";
    public static final String SCORES = "scores";
    public static final String POSITIVE_THRESHOLDS = "positiveThresholds";

    /**
     * Load data file in LibSVm format. The documents IDs are assigned according to the row index in the original
     * file, i.e. useful at classification time. We are assuming that the feature IDs are the same as the training
     * file used to build the classification model.
     *
     * @param sc       The spark context.
     * @param dataFile The data file.
     * @return An RDD containing the read points.
     */
    public static JavaRDD<MultilabelPoint> loadLibSvmFileFormatDataAsList(JavaSparkContext sc, String dataFile, boolean labels0Based, boolean binaryProblem) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");

        JavaRDD<String> lines = sc.textFile(dataFile).cache();
        int numFeatures = computeNumFeaturesFromLibSvmFormat(lines);

        ArrayList<MultilabelPoint> points = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(dataFile));

            try {
                int docID = 0;
                String line = br.readLine();
                while (line != null) {
                    if (line.isEmpty())
                        return null;
                    String[] fields = line.split("\\s+");
                    String[] t = fields[0].split(",");
                    int[] labels = new int[0];
                    if (!binaryProblem) {
                        labels = new int[t.length];
                        for (int i = 0; i < t.length; i++) {
                            String label = t[i];
                            if (labels0Based)
                                labels[i] = new Double(Double.parseDouble(label)).intValue();
                            else
                                labels[i] = new Double(Double.parseDouble(label)).intValue() - 1;
                            if (labels[i] < 0)
                                throw new IllegalArgumentException("In current configuration I obtain a negative label ID value. Please check if this is a problem binary or multiclass " +
                                        "and if the labels IDs are in form 0-based or 1-based");
                        }
                    } else {
                        if (t.length > 1)
                            throw new IllegalArgumentException("In binary problem you can only specify one label ID (+1 or -1) per document as valid label IDs");
                        int label = new Double(Double.parseDouble(t[0])).intValue();
                        if (label > 0) {
                            labels = new int[]{0};
                        }
                    }
                    ArrayList<Integer> indexes = new ArrayList<Integer>();
                    ArrayList<Double> values = new ArrayList<Double>();
                    for (int j = 1; j < fields.length; j++) {
                        String data = fields[j];
                        if (data.startsWith("#"))
                            // Beginning of a comment. Skip it.
                            break;
                        String[] featInfo = data.split(":");
                        // Transform feature ID value in 0-based.
                        int featID = Integer.parseInt(featInfo[0]) - 1;
                        double value = Double.parseDouble(featInfo[1]);
                        indexes.add(featID);
                        values.add(value);
                    }

                    points.add(new MultilabelPoint(docID, numFeatures, indexes.stream().mapToInt(i -> i).toArray(), values.stream().mapToDouble(i -> i).toArray(), labels));

                    line = br.readLine();
                    docID++;
                }
            } finally {
                br.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Reading input LibSVM data file", e);
        }

        return sc.parallelize(points);
    }

    /**
     * Load data file in LibSVm format. The documents IDs are assigned arbitrarily by Spark.
     *
     * @param sc       The spark context.
     * @param dataFile The data file.
     * @return An RDD containing the read points.
     */
    public static JavaRDD<MultilabelPoint> loadLibSvmFileFormatData(JavaSparkContext sc, String dataFile, boolean labels0Based, boolean binaryProblem) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");
        JavaRDD<String> lines = sc.textFile(dataFile).cache();
        int localNumFeatures = computeNumFeaturesFromLibSvmFormat(lines);
        Broadcast<Integer> distNumFeatures = sc.broadcast(localNumFeatures);
        JavaRDD<MultilabelPoint> docs = lines.filter(line -> !line.isEmpty()).zipWithIndex().map(item -> {
            int numFeatures = distNumFeatures.getValue();
            String line = item._1();
            long indexLong = item._2();
            int index = (int) indexLong;
            String[] fields = line.split("\\s+");
            String[] t = fields[0].split(",");
            int[] labels = new int[0];
            if (!binaryProblem) {
                labels = new int[t.length];
                for (int i = 0; i < t.length; i++) {
                    String label = t[i];
                    // Labels should be already 0-based.
                    if (labels0Based)
                        labels[i] = new Double(Double.parseDouble(label)).intValue();
                    else
                        labels[i] = new Double(Double.parseDouble(label)).intValue() - 1;
                    if (labels[i] < 0)
                        throw new IllegalArgumentException("In current configuration I obtain a negative label ID value. Please check if this is a problem binary or multiclass " +
                                "and if the labels IDs are in form 0-based or 1-based");
                    assert (labels[i] >= 0);
                }
            } else {
                if (t.length > 1)
                    throw new IllegalArgumentException("In binary problem you can only specify one label ID (+1 or -1) per document as valid label IDs");
                int label = new Double(Double.parseDouble(t[0])).intValue();
                if (label > 0) {
                    labels = new int[]{0};
                }
            }
            ArrayList<Integer> indexes = new ArrayList<Integer>();
            ArrayList<Double> values = new ArrayList<Double>();
            for (int j = 1; j < fields.length; j++) {
                String data = fields[j];
                if (data.startsWith("#"))
                    // Beginning of a comment. Skip it.
                    break;
                String[] featInfo = data.split(":");
                // Transform feature ID value in 0-based.
                int featID = Integer.parseInt(featInfo[0]) - 1;
                double value = Double.parseDouble(featInfo[1]);
                indexes.add(featID);
                values.add(value);
            }

            return new MultilabelPoint(index, numFeatures, indexes.stream().mapToInt(i -> i).toArray(), values.stream().mapToDouble(i -> i).toArray(), labels);
        });

        return docs;
    }

    /**
     * Compute the number of distinct features used in the specified RDD. Each line in the RDD must
     * be in LibSvm format.
     *
     * @param lines The dataset to be analyzed.
     * @return The number of distinct features found on RDD. The valid features IDs will be in the
     * range [0, numFeatures-1].
     */
    public static int computeNumFeaturesFromLibSvmFormat(JavaRDD<String> lines) {
        int maxFeatureID = lines.map(line -> {
            if (line.isEmpty())
                return -1;
            String[] fields = line.split("\\s+");
            int maximumFeatID = 0;
            for (int j = 1; j < fields.length; j++) {
                String data = fields[j];
                if (data.startsWith("#"))
                    // Beginning of a comment. Skip it.
                    break;
                String[] featInfo = data.split(":");
                int featID = Integer.parseInt(featInfo[0]);
                maximumFeatID = Math.max(featID, maximumFeatID);
            }
            return maximumFeatID;
        }).reduce((val1, val2) -> val1 < val2 ? val2 : val1);

        return maxFeatureID;
    }


    /**
     * Compute the number of distinct features used in the specified RDD. The data contained in
     * column {@code fieldName} must be in the format as coded in {@link #multilabelPointDataType()}.
     *
     * @param rows      The RDD to be analyzed.
     * @param fieldName The column containing the interesting data.
     * @return The number of distict features used. The valid features IDs will be in the
     * range [0, numFeatures-1].
     */
    public static int computeNumFeaturesFromDataFrame(JavaRDD<Row> rows, String fieldName) {
        int maxFeatureID = rows.map(row -> {
            Row rowFeatures = row.getStruct(row.fieldIndex(fieldName));
            List<Integer> features = rowFeatures.getList(rowFeatures.fieldIndex(FEATURES));
            int maximumFeatID = 0;
            for (int featureID : features) {
                if (featureID > maximumFeatID)
                    maximumFeatID = featureID;
            }
            return maximumFeatID;
        }).reduce((val1, val2) -> val1 < val2 ? val2 : val1);

        return maxFeatureID;
    }


    public static int getNumDocuments(JavaRDD<MultilabelPoint> documents) {
        if (documents == null)
            throw new NullPointerException("The documents RDD is 'null'");
        return (int) documents.count();
    }

    public static int getNumLabels(JavaRDD<MultilabelPoint> documents) {
        if (documents == null)
            throw new NullPointerException("The documents RDD is 'null'");
        int maxValidLabelID = documents.map(doc -> {
            List<Integer> values = Arrays.asList(ArrayUtils.toObject(doc.getLabels()));
            if (values.size() == 0)
                return 0;
            else
                return Collections.max(values);
        }).reduce((m1, m2) -> Math.max(m1, m2));
        return maxValidLabelID + 1;
    }

    public static int getNumFeatures(JavaRDD<MultilabelPoint> documents) {
        if (documents == null)
            throw new NullPointerException("The documents RDD is 'null'");
        return documents.take(1).get(0).getNumFeatures();
    }

    public static JavaRDD<LabelDocuments> getLabelDocuments(JavaRDD<MultilabelPoint> documents) {
        return documents.flatMapToPair(doc -> {
            int[] labels = doc.getLabels();
            ArrayList<Integer> docAr = new ArrayList<Integer>();
            docAr.add(doc.getPointID());
            ArrayList<Tuple2<Integer, ArrayList<Integer>>> ret = new ArrayList<Tuple2<Integer, ArrayList<Integer>>>();
            for (int i = 0; i < labels.length; i++) {
                ret.add(new Tuple2<>(labels[i], docAr));
            }
            return ret;
        }).reduceByKey((list1, list2) -> {
            ArrayList<Integer> ret = new ArrayList<Integer>();
            ret.addAll(list1);
            ret.addAll(list2);
            Collections.sort(ret);
            return ret;
        }).map(item -> {
            return new LabelDocuments(item._1(), item._2().stream().mapToInt(i -> i).toArray());
        });
    }

    public static JavaRDD<FeatureDocuments> getFeatureDocuments(JavaRDD<MultilabelPoint> documents) {
        return documents.flatMapToPair(doc -> {
            SparseVector feats = doc.getFeaturesAsSparseVector();
            int[] indices = feats.indices();
            ArrayList<Tuple2<Integer, FeatureDocuments>> ret = new ArrayList<>();
            for (int i = 0; i < indices.length; i++) {
                int featureID = indices[i];
                int[] docs = new int[]{doc.getPointID()};
                int[][] labels = new int[1][];
                labels[0] = doc.getLabels();
                ret.add(new Tuple2<>(featureID, new FeatureDocuments(featureID, docs, labels)));
            }
            return ret;
        }).reduceByKey((f1, f2) -> {
            int numDocs = f1.getDocuments().length + f2.getDocuments().length;
            int[] docsMerged = new int[numDocs];
            int[][] labelsMerged = new int[numDocs][];
            // Add first feature info.
            for (int idx = 0; idx < f1.getDocuments().length; idx++) {
                docsMerged[idx] = f1.getDocuments()[idx];
            }
            for (int idx = 0; idx < f1.getDocuments().length; idx++) {
                labelsMerged[idx] = f1.getLabels()[idx];
            }

            // Add second feature info.
            for (int idx = f1.getDocuments().length; idx < numDocs; idx++) {
                docsMerged[idx] = f2.getDocuments()[idx - f1.getDocuments().length];
            }
            for (int idx = f1.getDocuments().length; idx < numDocs; idx++) {
                labelsMerged[idx] = f2.getLabels()[idx - f1.getDocuments().length];
            }
            return new FeatureDocuments(f1.featureID, docsMerged, labelsMerged);
        }).map(item -> item._2());
    }


    public static DataType multilabelPointDataType() {
        List<StructField> fields = new ArrayList<>();
        fields.add(DataTypes.createStructField(POINT_ID, DataTypes.IntegerType, false));
        fields.add(DataTypes.createStructField(FEATURES, DataTypes.createArrayType(DataTypes.IntegerType, false), false));
        fields.add(DataTypes.createStructField(WEIGHTS, DataTypes.createArrayType(DataTypes.DoubleType, false), false));
        fields.add(DataTypes.createStructField(LABELS, DataTypes.createArrayType(DataTypes.IntegerType, false), false));
        StructType st = DataTypes.createStructType(fields);
        return st;
    }


    public static Row fromMultilabelPoint(MultilabelPoint pt) {
        Cond.requireNotNull(pt, "pt");
        return RowFactory.create(pt.getPointID(), pt.getFeatures(), pt.getWeights(), pt.getLabels());
    }


    public static MultilabelPoint toMultilabelPoint(Row row, String fieldName, int numFeatures) {
        Cond.requireNotNull(row, "row");
        Cond.requireNotNull(fieldName, "fieldName");
        Cond.require(!fieldName.isEmpty(), "The field name is empty");
        int idx = row.fieldIndex(fieldName);
        Cond.require(idx >= 0, "The requested field name <" + fieldName + "> is not available");
        Row inputPoint = row.getStruct(idx);

        int pointID = inputPoint.getInt(inputPoint.fieldIndex(POINT_ID));
        int[] features = toIntArray(inputPoint.getList(inputPoint.fieldIndex(FEATURES)));
        double[] weights = toDoubleArray(inputPoint.getList(inputPoint.fieldIndex(WEIGHTS)));
        int[] labels = toIntArray(inputPoint.getList(inputPoint.fieldIndex(LABELS)));
        MultilabelPoint point = new MultilabelPoint(pointID, numFeatures, features, weights, labels);
        return point;
    }

    public static DataType pointClassificationResultsDataType() {
        List<StructField> fields = new ArrayList<>();
        fields.add(DataTypes.createStructField(POINT_ID, DataTypes.IntegerType, false));
        fields.add(DataTypes.createStructField(LABELS, DataTypes.createArrayType(DataTypes.IntegerType, false), false));
        fields.add(DataTypes.createStructField(SCORES, DataTypes.createArrayType(DataTypes.DoubleType, false), false));
        fields.add(DataTypes.createStructField(POSITIVE_THRESHOLDS, DataTypes.createArrayType(DataTypes.DoubleType, false), false));
        StructType st = DataTypes.createStructType(fields);
        return st;
    }

    /**
     * Check if the specified data type have a structure compatible with a multilabel point data type. If the
     * structure is not compatible, the method will raise an exception.
     *
     * @param dt The data type to be checked.
     */
    public static void checkMultilabelPointDataType(StructType dt) {
        Cond.require(dt.fieldIndex(POINT_ID) >= 0, "The field " + POINT_ID + " does not exist!");
        Cond.require(dt.fieldIndex(FEATURES) >= 0, "The field " + FEATURES + " does not exist!");
        Cond.require(dt.fieldIndex(WEIGHTS) >= 0, "The field " + WEIGHTS + " does not exist!");
        Cond.require(dt.fieldIndex(LABELS) >= 0, "The field " + LABELS + " does not exist!");
    }


    /**
     * Convert the row data contained in {@code fieldName} field to an instance of class {@link PointClassificationResults}.
     * The column data must be in format as specified in {@link #pointClassificationResultsDataType()} code.
     *
     * @param row       The row containing data to be converted.
     * @param fieldName The field name inside the row.
     * @return A corresponding instance.
     */
    public static PointClassificationResults toPointClassificationResults(Row row, String fieldName) {
        Cond.requireNotNull(row, "row");
        Cond.requireNotNull(fieldName, "fieldName");
        Cond.require(!fieldName.isEmpty(), "fieldName is empty");
        int idx = row.fieldIndex(fieldName);
        Cond.require(idx >= 0, "The requested field name <" + fieldName + "> is not available");
        Row clResults = row.getStruct(idx);
        int pointID = clResults.getInt(clResults.fieldIndex(DataUtils.POINT_ID));
        int[] labels = DataUtils.toIntArray(clResults.getList(clResults.fieldIndex(DataUtils.LABELS)));
        double[] scores = DataUtils.toDoubleArray(clResults.getList(clResults.fieldIndex(DataUtils.SCORES)));
        double[] positiveThreshold = DataUtils.toDoubleArray(clResults.getList(clResults.fieldIndex(DataUtils.POSITIVE_THRESHOLDS)));
        return new PointClassificationResults(pointID, labels, scores, positiveThreshold);
    }


    /**
     * Convert the specified list of integers to a native array.
     *
     * @param l The list to convert.
     * @return The resulting array.
     */
    public static int[] toIntArray(List<Integer> l) {
        Cond.requireNotNull(l, "l");
        int[] ret = new int[l.size()];
        for (int i = 0; i < l.size(); i++)
            ret[i] = l.get(i);
        return ret;
    }


    /**
     * Convert the specified list of doubles to a native array.
     *
     * @param l The list to convert.
     * @return The resulting array.
     */
    public static double[] toDoubleArray(List<Double> l) {
        Cond.requireNotNull(l, "l");
        double[] ret = new double[l.size()];
        for (int i = 0; i < l.size(); i++)
            ret[i] = l.get(i);
        return ret;
    }




    public static class MultilabelPointFieldMapping implements Serializable {
        private final String pointIDField;
        private final String featuresField;
        private final String weightsField;
        private final String labelsField;

        public MultilabelPointFieldMapping(String pointIDField, String featuresField, String weightsField, String labelsField) {
            Cond.requireNotNull(pointIDField, "pointIDField");
            Cond.requireNotNull(featuresField, "featuresField");
            Cond.requireNotNull(weightsField, "weightsField");
            Cond.requireNotNull(labelsField, "labelsField");
            this.pointIDField = pointIDField;
            this.featuresField = featuresField;
            this.weightsField = weightsField;
            this.labelsField = labelsField;
        }

        public String getPointIDField() {
            return pointIDField;
        }

        public String getFeaturesField() {
            return featuresField;
        }

        public String getWeightsField() {
            return weightsField;
        }

        public String getLabelsField() {
            return labelsField;
        }
    }


    /**
     * Create a new dataframe corresponding to the set of specified text documents. The name
     * of the fieds in generated data frame correspond to the name of the private fields declared in the class
     * {@link TextualDocumentWithLabels} and its ancestors.
     *
     * @param docs The set of documents to import in the created dataframe.
     * @return The generated dataframe.
     */
    public static DataFrame toTextualDocumentWithlabelsDataFrame(JavaRDD<TextualDocumentWithLabels> docs) {
        Cond.requireNotNull(docs, "docs");
        SQLContext sqlContext = new SQLContext(docs.context());
        return sqlContext.createDataFrame(docs, TextualDocumentWithLabels.class);
    }


    /**
     * Create a new dataframe corresponding to the set of specified text documents. The name
     * of the fieds in generated data frame correspond to the name of the private fields declared in the class
     * {@link TextualDocumentWithLabels} and its ancestors.
     *
     * @param docs The set of documents to import in the created dataframe.
     * @return The generated dataframe.
     */
    public static DataFrame toTextualDocumentDataFrame(JavaRDD<TextualDocument> docs) {
        Cond.requireNotNull(docs, "docs");
        SQLContext sqlContext = new SQLContext(docs.context());
        return sqlContext.createDataFrame(docs, TextualDocument.class);
    }

    /**
     * Append a column with multilabel point schema starting from a source data frame and a
     * mapping of the available fields in the source data frame with the semantic fields in class
     * {@link MultilabelPoint}.
     *
     * @param df The source data frame.
     * @param multilabelPointFieldName The name of new column to append to the end of source data frame.
     * @param fi The mapping for fields in the source data frame.
     * @return A new data frame with the new column added at the end of all available columns.
     */
    public static DataFrame toMultilabelPointDataFrame(DataFrame df, String multilabelPointFieldName, MultilabelPointFieldMapping fi) {
        Cond.requireNotNull(df, "df");
        Cond.requireNotNull(fi, "fi");
        Cond.requireNotNull(multilabelPointFieldName, "multilabelPointFieldName");
        JavaSparkContext sc = new JavaSparkContext(df.sqlContext().sparkContext());
        Broadcast<MultilabelPointFieldMapping> fiBr = sc.broadcast(fi);
        JavaRDD<Row> updatedRows = df.toJavaRDD().map(row -> {
            MultilabelPointFieldMapping localFi = fiBr.value();
            Object[] values = new Object[row.size()+1];
            for (int i = 0; i < row.size(); i++) {
                values[i] = row.get(i);
            }
            int pointID = row.getInt(row.fieldIndex(localFi.getPointIDField()));
            int[] features = toIntArray(row.getList(row.fieldIndex(localFi.getFeaturesField())));
            double[] weights = toDoubleArray(row.getList(row.fieldIndex(localFi.getWeightsField())));
            int[] labels = toIntArray(row.getList(row.fieldIndex(localFi.getLabelsField())));
            Row pt = RowFactory.create(pointID, features, weights, labels);
            values[values.length-1] = pt;
            return RowFactory.create(values);
        });

        // Update schema.
        StructType oldSchema = df.schema();
        List<StructField> fields = new ArrayList<>();
        for (int i = 0; i < oldSchema.fields().length; i++) {
            fields.add(oldSchema.fields()[i]);
        }
        DataType outDataType = DataUtils.multilabelPointDataType();
        fields.add(DataTypes.createStructField(multilabelPointFieldName, outDataType, false));
        StructType updatedSchema = DataTypes.createStructType(fields);

        // Create new data frame.
        DataFrame dfRet = df.sqlContext().createDataFrame(updatedRows, updatedSchema);
        return dfRet;
    }


    public static Object[] copyValuesFromRow(Row row, int numAdditionalEmptyFields) {
        Cond.requireNotNull(row, "row");
        Cond.require(numAdditionalEmptyFields >= 0, "The number of additional fields must be greater equals than 0");
        Object[] fields = new Object[row.length()+numAdditionalEmptyFields];
        for (int i = 0; i < row.length(); i++)
            fields[i] = row.get(i);
        return fields;
    }


    public static class LabelDocuments implements Serializable {
        private final int labelID;
        private final int[] documents;

        public LabelDocuments(int labelID, int[] documents) {
            this.labelID = labelID;
            this.documents = documents;
        }

        public int getLabelID() {
            return labelID;
        }

        public int[] getDocuments() {
            return documents;
        }
    }

    public static class FeatureDocuments implements Serializable {
        private final int featureID;
        private final int[] documents;
        private final int[][] labels;

        public FeatureDocuments(int featureID, int[] documents, int[][] labels) {
            this.featureID = featureID;
            this.documents = documents;
            this.labels = labels;
        }

        public int getFeatureID() {
            return featureID;
        }

        public int[] getDocuments() {
            return documents;
        }

        public int[][] getLabels() {
            return labels;
        }
    }
}
