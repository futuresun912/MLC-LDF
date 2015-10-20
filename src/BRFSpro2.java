import meka.classifiers.multilabel.BR;
import meka.core.MLUtils;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by sunlu on 10/21/15.
 * Directly applied with two-stage FS
 * Incomplete
 */
public class BRFSpro2 extends BR {

    protected int[][] m_Indices1;
    protected int[][] m_Indices2;
    protected int[][] m_Indices;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int L = D.classIndex();

        m_MultiClassifiers = AbstractClassifier.makeCopies(m_Classifier, L);
        m_InstancesTemplates = new Instances[L];
        m_Indices1 = new int[L][];
        m_Indices2 = new int[L][];


        for(int j = 0; j < L; j++) {
            Instances D_j = MLUtils.keepAttributesAt(new Instances(D), new int[]{j}, L);
            D_j.setClassIndex(0);

            // First-stage FS
            AttributeSelection selector1 = new AttributeSelection();
            CfsSubsetEval evaluator1 = new CfsSubsetEval();
            GreedyStepwise searcher1 = new GreedyStepwise();
            searcher1.setNumExecutionSlots(8);
            selector1.setEvaluator(evaluator1);
            selector1.setSearch(searcher1);

            selector1.SelectAttributes(D_j);
            m_Indices1[j] = selector1.selectedAttributes();
            D_j = selector1.reduceDimensionality(D_j);
//            D_j.setClassIndex(0);

            // Second-stage FS
            AttributeSelection selector2 = new AttributeSelection();
            WrapperSubsetEval evaluator2 = new WrapperSubsetEval();
            GreedyStepwise searcher2 = new GreedyStepwise();
            searcher2.setNumExecutionSlots(8);
            selector2.setEvaluator(evaluator2);
            selector2.setSearch(searcher2);

            selector2.SelectAttributes(D_j);
            m_Indices2[j] = selector2.selectedAttributes();
            D_j = selector2.reduceDimensionality(D_j);
//            D_j.setClassIndex(0);

            m_MultiClassifiers[j].buildClassifier(D_j);
            m_InstancesTemplates[j] = new Instances(D_j, 0);
        }
    }

    @Override
    public double[] distributionForInstance(Instance x) throws Exception {

        int L = x.classIndex();
        double y[] = new double[L];

        for (int j = 0; j < L; j++) {
            Instance x_j = (Instance)x.copy();
            x_j.setDataset(null);
            x_j = MLUtils.keepAttributesAt(x_j,new int[]{j},L);
            x_j = MLUtils.keepAttributesAt(x_j,m_Indices1[j],x_j.numAttributes());
            x_j = MLUtils.keepAttributesAt(x_j,m_Indices2[j],x_j.numAttributes());
            x_j.setDataset(m_InstancesTemplates[j]);
            y[j] = m_MultiClassifiers[j].distributionForInstance(x_j)[1];
        }

        return y;
    }

}
