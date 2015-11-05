
import meka.core.A;
import meka.core.MLUtils;
import sun.misc.Sort;
import weka.attributeSelection.*;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;

import java.util.Arrays;

/**
 * Created by sunlu on 9/14/15.
 * This class is designed to mine label-specific features by two-stage feature selection
 * It should implement the following methods:
 * 1. feaSelect1
 * 2. feaSelect2
 * 3. instTransform
 * 4. shiftIndices
 * 5. initialFS   /incomplete
 */
public class MLFeaSelect {

    protected int L;
    protected int m_numThreads;
    protected boolean m_FlagRanker;
    protected boolean[] m_FlagFS;
    protected boolean m_IG_On;
    protected boolean m_Cfs_On;
    protected double m_PercentFeature;
    protected int[][] m_Indices1;
    protected int[][] m_Indices2;
    protected int[][] m_Indices;
    protected Instances[] m_instHeader;
    protected boolean m_IGPercent;
//    protected int[] m_paPACC;

    public MLFeaSelect(int L) {
        this.L = L;
        this.m_numThreads = 8;
        this.m_FlagRanker = false;
        this.m_FlagFS = new boolean[2];
        this.m_IG_On = false;
        this.m_Cfs_On = false;
        Arrays.fill(this.m_FlagFS, false);
        this.m_PercentFeature = 0.2;
        this.m_Indices1 = new int[L][];
        this.m_Indices2 = new int[L][];
        this.m_Indices = new int[L][];
        this.m_instHeader = new Instances[L];
        this.m_IGPercent =false;
    }

//    protected int[] getPACCpa() throws Exception {
//        return this.m_paPACC;
//    }

    protected void setNumThreads(int numThreads) throws Exception {
        this.m_numThreads = numThreads;
    }

    protected void setPercentFeature(double fraction) throws Exception {
        this.m_PercentFeature = fraction;
        this.m_IGPercent = true;
    }


    protected void setFilterIG(boolean IG_On) throws Exception {
        this.m_IG_On = IG_On;
    }

    protected void setWrapperCfs(boolean Cfs_On) throws Exception {
        this.m_Cfs_On = Cfs_On;
    }

    // The first-stage feature selection for MLC
    protected Instances[] feaSelect1(Instances D) throws Exception {

        int n = D.numAttributes();
        Instances[] outputD = new Instances[L];

        if (!m_IG_On) {
            AttributeSelection selector;
            CfsSubsetEval evaluator;
            GreedyStepwise searcher;
//            BestFirst searcher;

            // Perform FS for each label
            for (int j = 0; j < L; j++) {

                // Remove all the labels except j
                int[] pa = new int[0];
                pa = A.append(pa, j);
                Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
                D_j.setClassIndex(0);

                // Initializing the feature selector
                selector = new AttributeSelection();
                evaluator = new CfsSubsetEval();

                // Greedy search
                searcher = new GreedyStepwise();
                searcher.setNumExecutionSlots(m_numThreads);
                searcher.setConservativeForwardSelection(true);

//                // BestFirst search
//                searcher = new BestFirst();
//                searcher.setSearchTermination(10);
//                searcher.setLookupCacheSize(5);

                selector.setEvaluator(evaluator);
                selector.setSearch(searcher);

                // Obtain the indices of selected features
                selector.SelectAttributes(D_j);
                m_Indices1[j] = selector.selectedAttributes();
                m_Indices1[j] = shiftIndices(m_Indices1[j], L, pa);

                D.setClassIndex(0);
                outputD[j] = MLUtils.keepAttributesAt(new Instances(D), m_Indices1[j], n);
                outputD[j].setClassIndex(L);
                D.setClassIndex(L);

                m_instHeader[j] = new Instances(outputD[j]);
                m_instHeader[j].delete();
                m_FlagRanker = false;
                System.out.println(j+" "+(outputD[j].numAttributes()-L));
            }
        } else  {
            int numFeature;
            if (m_IGPercent) {
                numFeature = (int) ((n - L) * m_PercentFeature);
            }
            else {
                numFeature = n > 30 ? 30 : n;
            }

            AttributeSelection selector;
            InfoGainAttributeEval evaluator;
            Ranker searcher;

            // Perform FS for each label
            for (int j = 0; j < L; j++) {

                // Remove all the labels except j
                int[] pa = new int[0];
                pa = A.append(pa, j);
                Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
                D_j.setClassIndex(0);

                selector = new AttributeSelection();
                evaluator = new InfoGainAttributeEval();
                searcher = new Ranker();
                searcher.setNumToSelect(numFeature);
                selector.setEvaluator(evaluator);
                selector.setSearch(searcher);

                // Obtain the indices of selected features
                selector.SelectAttributes(D_j);
                m_Indices1[j] = selector.selectedAttributes();
                // Sort the selected features for the Ranker
                m_FlagRanker = true;
                m_Indices1[j] = shiftIndices(m_Indices1[j], L, pa);

                D.setClassIndex(0);
                outputD[j] = MLUtils.keepAttributesAt(new Instances(D), m_Indices1[j], n);
                outputD[j].setClassIndex(L);
                D.setClassIndex(L);

                m_instHeader[j] = new Instances(outputD[j]);
                m_instHeader[j].delete();

                System.out.println(j+" "+(outputD[j].numAttributes()-L));
            }
        }

        System.out.println("********************************");
        m_FlagFS[0] = true;
        return outputD;
    }


    // The first-stage feature selection for MLC
    protected Instances[] feaSelect1IR(Instances D, double[] factor) throws Exception {

        int n = D.numAttributes();
        Instances[] outputD = new Instances[L];

        if (!m_IG_On) {
            AttributeSelection selector;
            CfsSubsetEval evaluator;
            GreedyStepwise searcher;

            // Perform FS for each label
            for (int j = 0; j < L; j++) {

                // Remove all the labels except j
                int[] pa = new int[0];
                pa = A.append(pa, j);
                Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
                D_j.setClassIndex(0);

                // Initializing the feature selector
                selector = new AttributeSelection();
                evaluator = new CfsSubsetEval();
                searcher = new GreedyStepwise();
                searcher.setNumExecutionSlots(m_numThreads);
                searcher.setConservativeForwardSelection(true);

                selector.setEvaluator(evaluator);
                selector.setSearch(searcher);

                // Obtain the indices of selected features
                selector.SelectAttributes(D_j);
                m_Indices1[j] = selector.selectedAttributes();
                // Sort the selected features for the Ranker
//            if (searcher instanceof Ranker)
//                m_FlagRanker = true;
                m_Indices1[j] = shiftIndices(m_Indices1[j], L, pa);

                D.setClassIndex(0);
                outputD[j] = MLUtils.keepAttributesAt(new Instances(D), m_Indices1[j], n);
                outputD[j].setClassIndex(L);
                D.setClassIndex(L);

                m_instHeader[j] = new Instances(outputD[j]);
                m_instHeader[j].delete();

                m_FlagRanker = false;
                System.out.println(j+" "+(outputD[j].numAttributes()-L));
            }
        } else  {
//            int numFeature;
//            if (m_IGPercent) {
//                numFeature = (int) ((n - L) * m_PercentFeature);
//            }
//            else {
//                numFeature = n > 30 ? 30 : n;
//            }

//            int numFeature = (int) ( (n-L) * factor[j]);

            AttributeSelection selector;
            InfoGainAttributeEval evaluator;
            Ranker searcher;

            // Perform FS for each label
            for (int j = 0; j < L; j++) {

                // Remove all the labels except j
                int[] pa = new int[0];
                pa = A.append(pa, j);
                Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
                D_j.setClassIndex(0);

                int numFeature = (int) ( (n-L) * factor[j]);
                selector = new AttributeSelection();
                evaluator = new InfoGainAttributeEval();
                searcher = new Ranker();
                searcher.setNumToSelect(numFeature);
                selector.setEvaluator(evaluator);
                selector.setSearch(searcher);

                // Obtain the indices of selected features
                selector.SelectAttributes(D_j);
                m_Indices1[j] = selector.selectedAttributes();
                // Sort the selected features for the Ranker
                m_FlagRanker = true;
                m_Indices1[j] = shiftIndices(m_Indices1[j], L, pa);

                D.setClassIndex(0);
                outputD[j] = MLUtils.keepAttributesAt(new Instances(D), m_Indices1[j], n);
                outputD[j].setClassIndex(L);
                D.setClassIndex(L);

                m_instHeader[j] = new Instances(outputD[j]);
                m_instHeader[j].delete();

                System.out.println(j+" "+(outputD[j].numAttributes()-L));
            }
        }

        System.out.println("********************************");
        m_FlagFS[0] = true;
        return outputD;
    }


    // The second-stage feature selection for MLC
    protected Instances feaSelect2(Instances D_j, int j, int[] pa, double factor) throws Exception {

        int n = D_j.numAttributes();

        // Remove all the labels except j and its parents
        D_j.setClassIndex(j);
        pa = A.append(pa, j);
        Instances tempD = MLUtils.keepAttributesAt(new Instances(D_j), pa, L);

        // Initialization of the feature selector
        AttributeSelection selector = new AttributeSelection();

        if (!m_Cfs_On) {
            // Wrapper evaluator
            WrapperSubset evaluator = new WrapperSubset();
            evaluator.setClassifier(new Logistic());
            evaluator.setFolds(5);
            evaluator.setIRfactor(factor);
            evaluator.setEvaluationMeasure(new SelectedTag(8, WrapperSubset.TAGS_EVALUATION));

            // GreedyStepwise search
            GreedyCC searcher = new GreedyCC();
            searcher.m_pa = pa;
            searcher.setNumExecutionSlots(m_numThreads);
            searcher.setConservativeForwardSelection(true);

//        // generate the start set for searching
//        int[] paIndices = Utils.sort(pa);
//        int[] paTemp = Arrays.copyOf(paIndices, paIndices.length - 1);
//        for (int k = 0; k < paTemp.length; k ++)
//            paTemp[k] += 1;
//        String startSet = Arrays.toString(paTemp).replace("[", "").replace("]","");
//        searcher.setStartSet(startSet);

            selector.setEvaluator(evaluator);
            selector.setSearch(searcher);
        } else {
            // Wrapper evaluator
            CfsSubsetEval evaluator = new CfsSubsetEval();

            // GreedyStepwise search
            GreedyCC searcher = new GreedyCC();
            searcher.m_pa = pa;
            searcher.setNumExecutionSlots(m_numThreads);
            searcher.setConservativeForwardSelection(true);

//            // BestFirst search
//            BestFirstLDF searcher = new BestFirstLDF();
//            searcher.m_pa = pa;
//            searcher.setSearchTermination(10);
//            searcher.setLookupCacheSize(5);

            selector.setEvaluator(evaluator);
            selector.setSearch(searcher);
        }

        // Obtain the indices of selected features
        selector.SelectAttributes(tempD);
        m_Indices2[j] = selector.selectedAttributes();
        m_Indices2[j] = shiftIndices(m_Indices2[j], L, pa);

        D_j.setClassIndex(0);
        Instances outputD = MLUtils.keepAttributesAt(new Instances(D_j), m_Indices2[j], n);
        outputD.setClassIndex(L);
        D_j.setClassIndex(L);

        // Save the header information for transform the test instance
        m_instHeader[j] = new Instances(outputD);
        m_instHeader[j].delete();

        System.out.println(j + " " + (outputD.numAttributes() - L));
        m_FlagFS[1] =true;
        return outputD;
    }



    // The second-stage feature selection for MLC
    protected Instances feaSelect2CFS(Instances D_j, int j, int[] pa, double factor) throws Exception {

        int n = D_j.numAttributes();
        pa = new int[0];

        // Remove all the labels except j and its parents
        D_j.setClassIndex(j);
        pa = A.append(pa, j);
        Instances tempD = MLUtils.keepAttributesAt(new Instances(D_j), pa, L);

        // Initialization of the feature selector
        AttributeSelection selector = new AttributeSelection();
        // Wrapper evaluator
        CfsSubsetEval evaluator = new CfsSubsetEval();

        // GreedyStepwise search
        GreedyStepwise searcher = new GreedyStepwise();
        searcher.setNumExecutionSlots(m_numThreads);
        searcher.setConservativeForwardSelection(true);

//        // BestFirst search
//        BestFirst searcher = new BestFirst();
//        searcher.setSearchTermination(10);
//        searcher.setLookupCacheSize(5);

        selector.setEvaluator(evaluator);
        selector.setSearch(searcher);

        // Obtain the indices of selected features
        selector.SelectAttributes(tempD);
        m_Indices2[j] = selector.selectedAttributes();
        m_Indices2[j] = shiftIndices(m_Indices2[j], L, pa);

        D_j.setClassIndex(0);
        Instances outputD = MLUtils.keepAttributesAt(new Instances(D_j), m_Indices2[j], n);
        outputD.setClassIndex(L);
        D_j.setClassIndex(L);

        // Save the header information for transform the test instance
        m_instHeader[j] = new Instances(outputD);
        m_instHeader[j].delete();

        System.out.println(j + " " + (outputD.numAttributes() - L));
        m_FlagFS[1] =true;
        return outputD;
    }




//    // The second-stage feature selection for MLC
//    protected Instances feaSelect2PACC(Instances D_j, int j, int[] paAll, int[] pa, double factor) throws Exception {
//
//        int n = D_j.numAttributes();
//        m_paPACC = pa.clone();
//
//        // Remove all the labels except j and its parents
//        D_j.setClassIndex(j);
//        paAll = A.append(paAll, j);
//        pa = A.append(pa, j);
//        Instances tempD = MLUtils.keepAttributesAt(new Instances(D_j), paAll, L);
//
//        // Initialization of the feature selector
//        AttributeSelection selector = new AttributeSelection();
//
//        // Wrapper evaluator
//        WrapperSubset evaluator = new WrapperSubset();
//        evaluator.setClassifier(new Logistic());
//        evaluator.setFolds(5);
//        evaluator.setIRfactor(factor);
//        evaluator.setEvaluationMeasure(new SelectedTag(8, WrapperSubset.TAGS_EVALUATION));
//
//        // GreedyStepwise search
//        GreedyCC searcher = new GreedyCC();
//        searcher.m_pa = pa;
//        searcher.setNumExecutionSlots(m_numThreads);
//        searcher.setConservativeForwardSelection(true);
//
////        // generate the start set for searching
////        int[] paIndices = Utils.sort(pa);
////        int[] paTemp = Arrays.copyOf(paIndices, paIndices.length - 1);
////        for (int k = 0; k < paTemp.length; k ++)
////            paTemp[k] += 1;
////        String startSet = Arrays.toString(paTemp).replace("[", "").replace("]","");
////        searcher.setStartSet(startSet);
//
//        selector.setEvaluator(evaluator);
//        selector.setSearch(searcher);
//
//        // Obtain the indices of selected features
//        selector.SelectAttributes(tempD);
//        m_Indices2[j] = selector.selectedAttributes();
//
//
//
//        if (paAll.length != pa.length) {
//            int[] tempIndex = {};
//            int[] tempK = {};
//            for (int k = 0; k < paAll.length ; k++) {
//                if (m_Indices2[j][k] < paAll.length) {
//                    for (int l : pa) {
//                        if (m_Indices2[j][k] != l) {     // remove paAll-pa
//                            tempIndex = A.append(tempIndex, m_Indices2[j][k]);
//                            tempK = A.append(tempK, k);
//                        }
//                    }
////                m_Indices2[j] = A.delete(m_Indices2[j],k);
//                }
//            }
//            m_Indices2[j] = A.delete(m_Indices2[j], tempK);
//            for (int k = 0; k < m_Indices2[j].length; k ++) {
//                if ( m_Indices2[j][k] == j) {
//                    m_Indices2[j] = A.delete(m_Indices2[j],k);
//                    m_Indices2[j] = A.append(m_Indices2[j],j);
//                    break;
//                }
//            }
//
//
//            int[] paIndex = Utils.sort(paAll);
//            for (int k : tempIndex) {
//                for (int l : paIndex) {
//                    if (k == l) {
//                        m_paPACC = A.append(m_paPACC, paAll[l]);
//                    }
//                }
//            }
//        }
//
//        m_Indices2[j] = shiftIndices(m_Indices2[j], L, paAll);
//
//        D_j.setClassIndex(0);
//        Instances outputD = MLUtils.keepAttributesAt(new Instances(D_j), m_Indices2[j], n);
//        outputD.setClassIndex(L);
//        D_j.setClassIndex(L);
//
//        // Save the header information for transform the test instance
//        m_instHeader[j] = new Instances(outputD);
//        m_instHeader[j].delete();
//
//        System.out.println(j + " " + (outputD.numAttributes() - L));
//
//        m_FlagFS[1] =true;
//        return outputD;
//    }




    // Transform an test instance based on the indices of selected features
    protected Instance[] instTransform(Instance x) throws Exception {

        int L = x.classIndex();
        int n = x.numAttributes();
        Instance[] outputX = new Instance[L];

        for (int j = 0; j < L; j ++) {

            Instance tempX = (Instance)x.copy();
            tempX.setDataset(null);

            if (m_FlagFS[0] && m_FlagFS[1]) {
                // m_Indices <-- m_Indices1( m_Indices2 )
                int[] indexTemp = new int[m_Indices2[j].length - L];
                for (int k = 0; k < m_Indices2[j].length - L; k++)
                    indexTemp[k] = m_Indices1[j][m_Indices2[j][k] - L];
                m_Indices[j] = indexTemp.clone();
                for (int k = 0; k < L; k++)
                    m_Indices[j] = A.append(m_Indices[j], k);
            } else if (m_FlagFS[0]) {
                m_Indices[j] = m_Indices1[j].clone();
            } else if (m_FlagFS[1]) {
                m_Indices[j] = m_Indices2[j].clone();
            }

            outputX[j] = MLUtils.keepAttributesAt(tempX, m_Indices[j], n);  // Consume too much time in this step
            outputX[j].setDataset(m_instHeader[j]);
        }

        return outputX;
    }


    // Shift the indices of selected features for filtering D_j
    protected int[] shiftIndices(int[] indices, int L, int[] pa) throws Exception {

        // Remove the current label j
        indices = A.delete(indices, indices.length-1);

        // remove the parent labels from m_Indices2[j] for post-processing
        if (pa.length > 1) {
            int[] indexTemp = new int[0];
            for (int j = 0; j < indices.length; j ++)
                if (indices[j] >= pa.length)
                    indexTemp = A.append(indexTemp, indices[j]);
            indices = indexTemp.clone();   // possible problem
        }

        // Shift indices of features by taking labels into account
        for (int j = 0; j < indices.length; j ++)
            indices[j] = indices[j] + L - pa.length;

        // Sort the feature indices in ascending order
        if (m_FlagRanker) {
            int[] temp1 = Utils.sort(indices);
            int[] temp2 = new int[indices.length];
            for (int j = 0; j < indices.length; j ++) {
                temp2[j] = indices[temp1[j]];
            }
            indices = temp2.clone();
        }

        // Add indices of labels
        for (int j = 0; j < L; j ++)
            indices = A.append(indices, j);

        return indices;
    }

}
