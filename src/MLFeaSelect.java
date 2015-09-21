
import meka.core.A;
import meka.core.MLUtils;
import weka.attributeSelection.*;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
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
    protected boolean[] m_Flag;
    protected double m_PercentFeature;
    protected int[][] m_Indices1;
    protected int[][] m_Indices2;
    protected int[][] m_Indices;
    protected Instances[] m_instHeader;

    public MLFeaSelect(int L) {
        this.L = L;
        this.m_numThreads = 1;
        this.m_FlagRanker = false;
        this.m_Flag = new boolean[2];
        Arrays.fill(this.m_Flag, false);
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
        GreedyCC searcher;
//        InfoGainAttributeEval evaluator;
//        Ranker searcher;
//        int numFeature = (int) ((n - L)*m_PercentFeature);

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
            searcher = new GreedyCC();
            searcher.m_pa = pa;
            searcher.setNumExecutionSlots(m_numThreads);

//            evaluator = new InfoGainAttributeEval();
//            searcher = new Ranker();
//            searcher.setNumToSelect(numFeature);

            selector.setEvaluator(evaluator);
            selector.setSearch(searcher);

            // Obtain the indices of selected features
            selector.SelectAttributes(D_j);
            m_Indices1[j] = selector.selectedAttributes();
            // Sort the selected features for the Ranker based searcher for instTransform
//            if (searcher instanceof Ranker)
//                m_FlagRanker = true;
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
        m_Flag[0] = true;
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

        // Initialization of the feature selector
        AttributeSelection selector = new AttributeSelection();

        // Correlation-based evaluator
//        CfsSubsetEval evaluator = new CfsSubsetEval();

        // Wrapper evaluator
        WrapperSubsetEval evaluator = new WrapperSubsetEval();
        evaluator.setClassifier(new Logistic());
        evaluator.setFolds(5);

        // BestFirst search
//        BestFirstFS searcher = new BestFirstFS();
//        searcher.m_pa = pa;
//        searcher.setSearchTermination(5);
//        searcher.setLookupCacheSize(3);

        // GreedyStepwise search
        GreedyCC searcher = new GreedyCC();
        searcher.m_pa = pa;
        searcher.setNumExecutionSlots(m_numThreads);

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

        m_Flag[1] =true;
        return outputD;
    }


    // Transform an test instance based on the indices of selected features
    protected Instance[] instTransform(Instance x) throws Exception {

        int L = x.classIndex();
        int n = x.numAttributes();
        Instance[] outputX = new Instance[L];

        for (int j = 0; j < L; j ++) {
            Instance tempX = (Instance)x.copy();
            tempX.setDataset(null);     // Can it be removed?

            if (m_Flag[0] == true && m_Flag[1] == true) {
                // m_Indices <-- m_Indices1( m_Indices2 )
                int[] indexTemp = new int[m_Indices2[j].length - L];
                for (int k = 0; k < m_Indices2[j].length - L; k++)
                    indexTemp[k] = m_Indices1[j][m_Indices2[j][k] - L];
                m_Indices[j] = indexTemp.clone();
                for (int k = 0; k < L; k++)
                    m_Indices[j] = A.append(m_Indices[j], k);
            }
            if (m_Flag[0] == true && m_Flag[1] ==false)
                m_Indices[j] = m_Indices1[j].clone();
            if (m_Flag[0] == false && m_Flag[1] == true)
                m_Indices[j] = m_Indices2[j].clone();

            outputX[j] = MLUtils.keepAttributesAt(tempX, m_Indices[j], n);
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
        if (m_FlagRanker == true) {
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
