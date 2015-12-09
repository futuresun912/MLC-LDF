/**
 * Created by sunlu on 12/3/15.
 * Sampling techniques can be used for efficient FS
 */

import meka.core.A;
import meka.core.F;
import meka.core.MLUtils;
import weka.attributeSelection.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.Arrays;


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
    protected Instances[] m_dataHeader;
    protected Instance[] m_instTemplate;

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
        this.m_dataHeader = new Instances[L];
        this.m_instTemplate = new Instance[L];
    }

    protected void setPercentFeature(double fraction) throws Exception {
        this.m_PercentFeature = fraction;
    }

    protected int getNumFeaCfs(int j) throws Exception {
        return m_Indices2[j].length;
    }


    // The first-stage feature selection for MLC
    protected void feaSelect1(Instances D, int num) throws Exception {

        int n = D.numAttributes();
        int numFeature;
        numFeature = (int) ((n - L) * m_PercentFeature);

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
    protected void feaSelect2(Instances D_j, int j) throws Exception {

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


    // Shift the indices of selected features for filtering D
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

        int[] index_j = new int[m_Indices[j].length+L];
        for (int k = 0; k < index_j.length; k ++) {
            if (k < L) {
                index_j[k] = k;
            }
            else {
                index_j[k] = m_Indices[j][k - L];
            }
        }

        D.setClassIndex(-1);
        Instances D_j = F.remove(new Instances(D), index_j, true);
        D_j.setClassIndex(L);
        D.setClassIndex(L);
        m_instTemplate[j] = D_j.instance(1);
        m_dataHeader[j] = new Instances(D_j, 0);
        return D_j;
    }

    // Transform an test instance based on the indices of selected features
    protected Instance instTransform(Instance x_j, int j) throws Exception {
        int L = x_j.classIndex();
        Instance x_out = m_instTemplate[j];
        copyInstValues(x_out, x_j, m_Indices[j], L);
        x_out.setDataset(m_dataHeader[j]);
        return x_out;
    }


    /**
     * CopyValues - Set x_dest[i++] = x_src[j] for all j in indices[].
     */
    public Instance copyInstValues(Instance x_dest, Instance x_src,
                                   int indices[], int L) {
        int i = L;
        for(int j : indices) {
            x_dest.setValue(i++,x_src.value(j));
        }
        return x_dest;
    }

}

