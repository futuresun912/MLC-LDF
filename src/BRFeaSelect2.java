/**
 * Created by sunlu on 10/21/15.
 * For BRFSpro
 * Gfs + Wrapper
 */

import meka.core.A;
import meka.core.MLUtils;
import weka.attributeSelection.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.core.*;

import java.util.Arrays;


public class BRFeaSelect2 {

    protected int L;
    protected int m_numThreads;
    protected boolean m_FlagRanker;
    protected boolean[] m_FlagFS;
    protected double m_PercentFeature;
    protected int[][] m_Indices1;
    protected int[][] m_Indices2;
    protected int[][] m_Indices;
    protected Instances[] m_instHeader;

    public BRFeaSelect2(int L) {
        this.L = L;
        this.m_numThreads = 1;
        this.m_FlagRanker = false;
        this.m_FlagFS = new boolean[2];
        Arrays.fill(this.m_FlagFS, false);
        this.m_PercentFeature = 0.2;
        this.m_Indices1 = new int[L][];
        this.m_Indices2 = new int[L][];
        this.m_Indices = new int[L][];
        this.m_instHeader = new Instances[L];
    }

    protected void setPercentFeature(double fraction) throws Exception {
        this.m_PercentFeature = fraction;
    }

    protected void setNumThreads(int numThreads) throws Exception {
        this.m_numThreads = numThreads;
    }

    // The first-stage feature selection for MLC
    protected Instances[] feaSelect1(Instances D) throws Exception {

        int n = D.numAttributes();
        Instances[] outputD = new Instances[L];
        AttributeSelection selector;
        CfsSubsetEval evaluator;
//        GreedyStepwise searcher;
//        BestFirst searcher;

        // Perform FS for each label
        for (int j = 0; j < L; j ++) {

            // Remove all the labels except j
            int[] pa = new int[0];
            pa = A.append(pa, j);
            Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
            D_j.setClassIndex(0);

            // Initializing the feature selector
            selector = new AttributeSelection();
            evaluator = new CfsSubsetEval();

            GreedyStepwise searcher = new GreedyStepwise();
            searcher.setNumExecutionSlots(m_numThreads);
//            searcher.setSearchBackwards(true);

//            searcher = new BestFirst();
//            searcher.setLookupCacheSize(10);
//            searcher.setSearchTermination(5);
//            searcher.setDirection(new SelectedTag(0, BestFirst.TAGS_SELECTION));

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

            System.out.println(j+" "+(outputD[j].numAttributes()-L));

        }

        System.out.println("*****************************************");
        m_FlagFS[0] = true;
        m_FlagRanker = false;
        return outputD;
    }


    // The second-stage feature selection for MLC
    protected Instances feaSelect2(Instances D_j, int j, int[] pa) throws Exception {

        int n = D_j.numAttributes();

        // Remove all the labels except j and its parents
        D_j.setClassIndex(j);
        pa = A.append(pa, j);
        Instances tempD = MLUtils.keepAttributesAt(new Instances(D_j), pa, L);
        tempD.setClassIndex(0);

        // Initialization of the feature selector
        AttributeSelection selector = new AttributeSelection();

        // Wrapper evaluator
        WrapperSubset evaluator = new WrapperSubset();
//        WrapperSubsetEval evaluator = new WrapperSubsetEval();
//        evaluator.setClassifier(new Logistic());
        evaluator.setClassifier(new Logistic());
        evaluator.setFolds(5);
//        evaluator.setEvaluationMeasure(new SelectedTag(1,WrapperSubsetEval.TAGS_EVALUATION));

//        // GreedyStepwise search
//        GreedyCC searcher = new GreedyCC();
////        int[] paTemp = new int[]{-1};
////        searcher.m_pa = paTemp;
//        searcher.setNumExecutionSlots(m_numThreads);
//        searcher.setSearchBackwards(true);

        GreedyStepwise searcher = new GreedyStepwise();
//        searcher.setSearchBackwards(true);
        searcher.setNumExecutionSlots(m_numThreads);

//        BestFirst searcher = new BestFirst();
//        searcher.setLookupCacheSize(10);
//        searcher.setSearchTermination(5);
////        searcher.setDirection(new SelectedTag(0, BestFirst.TAGS_SELECTION));

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


    // Transform an test instance based on the indices of selected features
    protected Instance[] instTransform(Instance x) throws Exception {

        int L = x.classIndex();
        int n = x.numAttributes();
        Instance[] outputX = new Instance[L];
//        Instance[] tempX = new Instance[L];

        for (int j = 0; j < L; j ++) {

            Instance tempX = (Instance)x.copy();
            tempX.setDataset(null);

//            tempX[j] = (Instance)x.copy();
//            tempX[j].setDataset(null);

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

//            tempX[j] = MLUtils.keepAttributesAt(tempX[j], m_Indices[j], n);  // Consume too much time in this step
//            tempX[j].setDataset(m_instHeader[j]);
        }

        return outputX;
//        return tempX;
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
