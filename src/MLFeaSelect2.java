/**
 * Created by sunlu on 12/3/15.
 */

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
public class MLFeaSelect2 {

    protected int L;
    protected int m_numThreads;
    protected boolean m_FlagRanker;
    protected boolean m_IG;
    protected boolean[] m_CFS;
    protected double m_PercentFeature;
    protected int[][] m_Indices1;
    protected int[][] m_Indices2;
    protected int[][] m_Indices;
    protected Instances[] m_instHeader;
    protected boolean m_IGPercent;

    public MLFeaSelect2(int L) {
        this.L = L;
        this.m_numThreads = 8;
        this.m_FlagRanker = false;
        this.m_IG = false;
        this.m_CFS = new boolean[L];
        Arrays.fill(this.m_CFS, false);
        this.m_PercentFeature = 0.2;
        this.m_Indices1 = new int[L][];
        this.m_Indices2 = new int[L][];
        this.m_Indices = new int[L][];
        this.m_instHeader = new Instances[L];
        this.m_IGPercent =false;
    }

    protected void setNumThreads(int numThreads) throws Exception {
        this.m_numThreads = numThreads;
    }

    protected void setPercentFeature(double fraction) throws Exception {
        this.m_PercentFeature = fraction;
        this.m_IGPercent = true;
    }

    protected int getNumFeaCfs(int j) throws Exception {
        return m_Indices2[j].length;
    }


    // The first-stage feature selection for MLC
    protected void feaSelect1(Instances D, int num) throws Exception {

        int n = D.numAttributes();
        int numFeature;
        if (m_IGPercent) {
            numFeature = (int) ((n - L) * m_PercentFeature);
//            numFeature = numFeature > 80 ? 80 : numFeature < 20 ? 20 : numFeature;
        }
        else {
            numFeature = 30;
        }

        // Perform FS for each label
        for (int j = 0; j < num; j++) {

            int[] pa = new int[0];
            pa = A.append(pa, j);
            Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
            D_j.setClassIndex(0);

            AttributeSelection selector = new AttributeSelection();
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker searcher = new Ranker();
            searcher.setNumToSelect(numFeature);
            selector.setEvaluator(evaluator);
            selector.setSearch(searcher);

            // Obtain the indices of selected features
            selector.SelectAttributes(D_j);
            m_Indices1[j] = selector.selectedAttributes();
            // Sort the selected features for the Ranker
            m_FlagRanker = true;
            m_Indices1[j] = shiftIndices(m_Indices1[j], L, pa);
            System.out.println(j+" "+m_Indices1[j].length);
        }
        System.out.println("********************************");
        m_IG = true;
    }


    // The second-stage feature selection for MLC
    protected void feaSelect2CFS(Instances D_j, int j) throws Exception {

        // Remove all the labels except j and its parents
        int[] pa = new int[0];
        D_j.setClassIndex(j);
        pa = A.append(pa, j);
        Instances tempD = MLUtils.keepAttributesAt(new Instances(D_j), pa, L);

        // Initialization of the feature selector
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();

//        BestFirst searcher = new BestFirst();
//        searcher.setSearchTermination(10);
//        searcher.setLookupCacheSize(5);

        GreedyStepwise searcher = new GreedyStepwise();
        searcher.setNumExecutionSlots(m_numThreads);
        searcher.setSearchBackwards(true);

        selector.setEvaluator(evaluator);
        selector.setSearch(searcher);

        // Obtain the indices of selected features
        selector.SelectAttributes(tempD);
        m_Indices2[j] = selector.selectedAttributes();
        m_Indices2[j] = shiftIndices(m_Indices2[j], L, pa);
        System.out.println(j + " " + m_Indices2[j].length);
        m_CFS[j] = true;
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

        return indices;
    }

    // Transform an instances based on the indices of selected features
    protected Instances instTransform(Instances D, int j) throws Exception {

        int L = D.classIndex();
        int n = D.numAttributes();

        if (m_IG && m_CFS[j]) {
            // m_Indices <-- m_Indices1( m_Indices2 )
            if ( m_Indices2[j].length == 0 ) {
                m_Indices[j] = m_Indices1[j].clone();
            } else {
                int[] indexTemp = new int[m_Indices2[j].length];
                for (int k = 0; k < m_Indices2[j].length; k++)
                    indexTemp[k] = m_Indices1[j][m_Indices2[j][k]-L];
                m_Indices[j] = indexTemp.clone();
            }
        } else if (m_IG) {
            m_Indices[j] = m_Indices1[j].clone();
        } else if (m_CFS[j]) {
            m_Indices[j] = m_Indices2[j].clone();
        }

        int[] index_j = m_Indices[j].clone();
        // Add indices of labels
        for (int k = 0; k < L; k ++)
            index_j = A.append(index_j, k);

        D.setClassIndex(0);
        Instances D_j = MLUtils.keepAttributesAt(new Instances(D), index_j, n);
        D_j.setClassIndex(L);
        D.setClassIndex(L);
        m_instHeader[j] = new Instances(D_j);
        m_instHeader[j].delete();
        return D_j;
    }

    // Transform an test instance based on the indices of selected features
    protected Instance instTransform(Instance x_j, int j) throws Exception {

        int L = x_j.classIndex();
        int n = x_j.numAttributes();
        Instance tempX = (Instance)x_j.copy();
        tempX.setDataset(null);

        int[] indexTemp = m_Indices[j].clone();
        for (int k = 0; k < L; k ++)
            indexTemp = A.append(indexTemp, k);

        x_j = MLUtils.keepAttributesAt(tempX, indexTemp, n);  // Consume too much time in this step
        x_j.setDataset(m_instHeader[j]);

        return x_j;
    }

}

