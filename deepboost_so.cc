#include <map>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>
using namespace std;

extern "C"{
static const float kTolerance = 1e-7;
string loss_type="exponential";
float beta=0;
float lambda=0;
int tree_depth=1;

typedef int Feature;
typedef int Label;
typedef int NodeId;
typedef float Value;
typedef float Weight;

typedef struct Example {
  vector<Value> values;
  Label label;
  Weight weight;
} Example;

// A tree node.
typedef struct Node {
  vector<Example*> examples;  // Examples at this node.
  Feature split_feature;  // Split feature.
  Value split_value;  // Split value.
  NodeId left_child_id;  // Pointer to left child, if any.
  NodeId right_child_id;  // Pointer to right child, if any.
  Weight positive_weight;  // Total weight of positive examples at this node.
  Weight negative_weight;  // Total weight of negative examples at this node.
  bool leaf;  // Is this node is a leaf?
  int depth;  // Depth of the node in the tree. Root node has depth 0.
} Node;

// A tree is a vector of nodes.
typedef vector<Node> Tree;

// A model is a vector of (weight, tree) pairs, i.e., a weighted combination of
// trees.
typedef vector<pair<Weight, Tree> > Model;

static int num_features;
static int num_examples;
static float the_normalizer;
static bool is_initialized = false;

void initdeepboost(){
  
}

float ComplexityPenalty(int tree_size) {
  float rademacher =
      sqrt(((2 * tree_size + 1) * (log(num_features + 2) / log(2))*
            log(num_examples)) /
           num_examples);
  float ret=((lambda * rademacher + beta) * num_examples) /
         (2 * the_normalizer);
  return ret;
}

float Gradient(float wgtd_error, int tree_size, float alpha, int sign_edge) {
  // TODO(usyed): Can we make some mild assumptions and get rid of sign_edge?
  const float complexity_penalty = ComplexityPenalty(tree_size);
  const float edge = wgtd_error - 0.5;
  const int sign_alpha = (alpha >= 0) ? 1 : -1;
  if (fabs(alpha) > kTolerance) {
    return edge + sign_alpha * complexity_penalty;
  } else if (fabs(edge) <= complexity_penalty) {
    return 0;
  } else {
    return edge - sign_edge * complexity_penalty;
  }
}

void InitializeTreeData(const vector<Example*>& examples, float normalizer) {
  num_examples = examples.size();
  num_features = examples[0]->values.size();
  the_normalizer = normalizer;
  is_initialized = true;
}

Node MakeRootNode(const vector<Example*>& examples) {
  Node root;
  root.examples = examples;
  root.positive_weight = root.negative_weight = 0;
  int size=examples.size();
  for (int i=0;i<size;++i) {
    const Example& example=*(examples[i]);
    if (example.label == 1) {
      root.positive_weight += example.weight;
    } else {  // label == -1
      root.negative_weight += example.weight;
    }
  }
  root.leaf = true;
  root.depth = 0;
  return root;
}

map<Value, pair<Weight, Weight> > MakeValueToWeightsMap(const Node& node,
                                                       Feature feature) {
  map<Value, pair<Weight, Weight> > value_to_weights;
  int size=node.examples.size();
  for (int i=0;i<size;++i) {
      const Example& example=*(node.examples[i]);
	  if (example.label == 1) {
      value_to_weights[example.values[feature]].first += example.weight;
    } else {  // label = -1
      value_to_weights[example.values[feature]].second += example.weight;
    }
  }
  return value_to_weights;
}

void BestSplitValue(const map<Value, pair<Weight, Weight> >& value_to_weights, const Node& node, int tree_size, Value* split_value, float* delta_gradient) {
  *delta_gradient = 0;
  Weight left_positive_weight = 0, left_negative_weight = 0, right_positive_weight = node.positive_weight, right_negative_weight = node.negative_weight;
  float old_error = fmin(left_positive_weight + right_positive_weight,
                         left_negative_weight + right_negative_weight);
  float old_gradient = Gradient(old_error, tree_size, 0, -1);
  //int size=value_to_weights.size();
  map<Value,pair<Weight,Weight> >::const_iterator it;
  for (it=value_to_weights.begin();it!=value_to_weights.end();++it) {
	//map< Value, pair<Weight, Weight>>::value_type& elem=(*it);
    left_positive_weight += (*it).second.first;
    right_positive_weight -= (*it).second.first;
    left_negative_weight += (*it).second.second;
    right_negative_weight -= (*it).second.second;
    float new_error = fmin(left_positive_weight, left_negative_weight) +
                      fmin(right_positive_weight, right_negative_weight);
    float new_gradient = Gradient(new_error, tree_size + 2, 0, -1);
    if (fabs(new_gradient) - fabs(old_gradient) >
        *delta_gradient + kTolerance) {
      *delta_gradient = fabs(new_gradient) - fabs(old_gradient);
      *split_value = (*it).first;
    }
  }
}

void MakeChildNodes(Feature split_feature, Value split_value, Node* parent,
                    Tree* tree) {
  parent->split_feature = split_feature;
  parent->split_value = split_value;
  parent->leaf = false;
  Node left_child, right_child;
  left_child.depth = right_child.depth = parent->depth + 1;
  left_child.leaf = right_child.leaf = true;
  left_child.positive_weight = left_child.negative_weight =
      right_child.positive_weight = right_child.negative_weight = 0;
  int size=(parent->examples).size();
  for (int i=0;i<size;++i) {
	Example& example=*((parent->examples)[i]);
    Node* child;
    if (example.values[split_feature] <= split_value) {
      child = &left_child;
    } else {
      child = &right_child;
    }
    // TODO(usyed): Moving examples around is inefficient.
    child->examples.push_back(&example);
    if (example.label == 1) {
      child->positive_weight += example.weight;
    } else {  // label == -1
      child->negative_weight += example.weight;
    }
  }
  parent->left_child_id = tree->size();
  parent->right_child_id = tree->size() + 1;
  tree->push_back(left_child);
  tree->push_back(right_child);
}

Tree TrainTree(const vector<Example*>& examples) {
  Tree tree;
  tree.push_back(MakeRootNode(examples));
  NodeId node_id = 0;
  while (node_id < tree.size()) {
    Node& node = tree[node_id];  // TODO(usyed): Too bad this can't be const.
    Feature best_split_feature;
    Value best_split_value;
    float best_delta_gradient = 0;
    for (Feature split_feature = 0; split_feature < num_features;
         ++split_feature) {
      const map<Value, pair<Weight, Weight> > value_to_weights =
          MakeValueToWeightsMap(node, split_feature);
      Value split_value;
      float delta_gradient;
      BestSplitValue(value_to_weights, node, tree.size(), &split_value,
                     &delta_gradient);
      if (delta_gradient > best_delta_gradient + kTolerance) {
        best_delta_gradient = delta_gradient;
        best_split_feature = split_feature;
        best_split_value = split_value;
      }
    }
    if (node.depth < tree_depth && best_delta_gradient > kTolerance) {
      MakeChildNodes(best_split_feature, best_split_value, &node, &tree);
    }
    ++node_id;
  }
  return tree;
}

Label ClassifyExample(const Example& example, const Tree& tree) {
  const Node* node = &tree[0];
  while (node->leaf == false) {
    if (example.values[node->split_feature] <= node->split_value) {
      node = &tree[node->left_child_id];
    } else {
      node = &tree[node->right_child_id];
    }
  }
  if (node->positive_weight >= node->negative_weight) {
    return 1;
  } else {
    return -1;
  }
}



float EvaluateTreeWgtd(const vector<Example*>& examples, const Tree& tree) {
  float wgtd_error = 0;
  int size=examples.size();
  for (int i=0;i<size;++i) {
	const Example& example=*(examples[i]);
    if (ClassifyExample(example, tree) != example.label) {
      wgtd_error += example.weight;
    }
  }
  return wgtd_error;
}


float ComputeEta(float wgtd_error, float tree_size, float alpha) {
  wgtd_error = fmax(wgtd_error, kTolerance);  // Helps with division by zero.
  const float error_term =
      (1 - wgtd_error) * exp(alpha) - wgtd_error * exp(-alpha);
  const float complexity_penalty = ComplexityPenalty(tree_size);
  const float ratio = complexity_penalty / wgtd_error;
  float eta;
  if (fabs(error_term) <= 2 * complexity_penalty) {
    eta = -alpha;
  } else if (error_term > 2 * complexity_penalty) {
    eta = log(-ratio + sqrt(ratio * ratio + (1 - wgtd_error)/wgtd_error));
  } else {
    eta = log(ratio + sqrt(ratio * ratio + (1 - wgtd_error)/wgtd_error));
  }
  return eta;
}

// TODO(usyed): examples is passed by non-const reference because the example
// weights need to be changed. This is bad style.
void AddTreeToModel(vector<Example*>& examples, Model* model) {
  // Initialize normalizer
  float normalizer;
  if (model->empty()) {
    if (loss_type == "exponential") {
      normalizer = exp(1) * static_cast<float>(examples.size());
    } else if (loss_type == "logistic") {
      normalizer =
          static_cast<float>(examples.size()) / (log(2) * (1 + exp(-1)));
    }
  }
  //cout<<"normalizer:"<<normalizer<<endl;
  InitializeTreeData(examples, normalizer);
  int best_old_tree_idx = -1;
  float best_wgtd_error, wgtd_error, gradient, best_gradient = 0;

  // Find best old tree
  bool old_tree_is_best = false;
  for (int i = 0; i < model->size(); ++i) {
    const float alpha = (*model)[i].first;
    if (fabs(alpha) < kTolerance) continue;  // Skip zeroed-out weights.
    const Tree& old_tree = (*model)[i].second;
    wgtd_error = EvaluateTreeWgtd(examples, old_tree);
    int sign_edge = (wgtd_error >= 0.5) ? 1 : -1;
    gradient = Gradient(wgtd_error, old_tree.size(), alpha, sign_edge);
    if (fabs(gradient) >= fabs(best_gradient)) {
      best_gradient = gradient;
      best_wgtd_error = wgtd_error;
      best_old_tree_idx = i;
      old_tree_is_best = true;
    }
  }

  // Find best new tree
  Tree new_tree = TrainTree(examples);
  wgtd_error = EvaluateTreeWgtd(examples, new_tree);
  gradient = Gradient(wgtd_error, new_tree.size(), 0, -1);
  if (model->empty() || fabs(gradient) > fabs(best_gradient)) {
    best_gradient = gradient;
    best_wgtd_error = wgtd_error;
    old_tree_is_best = false;
  }

  // Update model weights
  float alpha;
  const Tree* tree;
  if (old_tree_is_best) {
    alpha = (*model)[best_old_tree_idx].first;
    tree = &((*model)[best_old_tree_idx].second);
  } else {
    alpha = 0;
    tree = &(new_tree);
  }
  const float eta = ComputeEta(best_wgtd_error, tree->size(), alpha);
  if (old_tree_is_best) {
    (*model)[best_old_tree_idx].first += eta;
  } else {
    model->push_back(make_pair(eta, new_tree));
  }

  // Update examples weights and compute normalizer
  const float old_normalizer = normalizer;
  normalizer = 0;
  int size=examples.size();
  for (int i=0;i<size;++i) {
	Example& example=*(examples[i]);
    const float u = eta * example.label * ClassifyExample(example, *tree);
    if (loss_type == "exponential") {
      example.weight *= exp(-u);
    } else if (loss_type == "logistic") {
      const float z = (1 - log(2) * example.weight * old_normalizer) /
                      (log(2) * example.weight * old_normalizer);
      example.weight = 1 / (log(2) * (1 + z * exp(u)));
    } 
    normalizer += example.weight;
  }

  // Renormalize example weights
  // TODO(usyed): Two loops is inefficient.
  size=examples.size();
  for (int i=0;i<size;++i) {
	Example& example=*(examples[i]);
    example.weight /= normalizer;
  }
}

/*Label ClassifyExample(const Example& example, const Model& model) {
  float score = 0;
  int size=model.size();
  for (int i=0;i<size;++i) {
	const pair<Weight,Tree>& wgtd_tree=model[i];
    score += wgtd_tree.first * ClassifyExample(example, wgtd_tree.second);
  }
  if (score < 0) {
    return -1;
  } else {
    return 1;
  }
}*/

float SoftClassifyExample(const Example& example, const Model& model) {
  float score = 0;
  int size=model.size();
  float weight=0;
  for (int i=0;i<size;++i) {
	const pair<Weight,Tree>& wgtd_tree=model[i];
    score += wgtd_tree.first * ClassifyExample(example, wgtd_tree.second);
	weight+=wgtd_tree.first;
  }
  return (weight!=0?score/weight:0);
}

void train(int r,int c, float* trainX, float* trainY, float _beta, float _lambda, int _tree_depth, int type,int round, Model** ret_model){
	beta=_beta;
	lambda=_lambda;
	tree_depth=_tree_depth;
	if (type==0) loss_type="exponential"; else loss_type="logistic";
	Model* model=new Model();
	vector<Example*> examples;
	for (int i=0;i<r;++i){
		Example* example=new Example();
		for (int j=0;j<c;++j) (example->values).push_back(trainX[i*c+j]);
		example->label=Label(trainY[i]);
		example->weight=1.0/r;
		examples.push_back(example);
	}
	for (int i=0;i<round;++i){
		AddTreeToModel(examples,model);
	}
	vector<Example*>().swap(examples);
	*ret_model=model;
}
	
void classify(int r,int c, float* testX, float* output,Model** _model){
	//Model model=**_model;
	for (int i=0;i<r;++i){
		Example example;
		for (int j=0;j<c;++j) (example.values).push_back(testX[i*c+j]);
		example.label=1;
		example.weight=1.0/r;
		output[i]=SoftClassifyExample(example, **_model);;
	}
}
void delete_model(Model** _model){
	delete *_model;
}
void feature_score(Model** _model,double* score){
	for (int i=0;i<num_features;++i)
		score[i]=0;
	int num_trees=(**_model).size();
	for (int i=0;i<num_trees;++i){
		float weight=(**_model)[i].first;
		Tree* tree=&((**_model)[i].second);
		Node* root=&((*tree)[0]);
		Node* right=&((*tree)[root->right_child_id]);
		float boost=(right->positive_weight)/(right->positive_weight+right->negative_weight)-(root->positive_weight)/(root->positive_weight+root->negative_weight);
		score[root->split_feature]+=boost*weight;
	}
	/*cout<<num_features<<endl;
	for (int i=0;i<num_features;++i)
		cout<<score[i]<<' ';
	cout<<endl;*/
}
}

/*int main(){
	beta=0;lambda=0;tree_depth=1;
	Model model;
	vector<Example> examples;
	Example examples_arr[5];
   	examples_arr[0].values = {1.0, 0.1, 11.0};
    examples_arr[0].label = 1;
    examples_arr[0].weight = 0.2;
    examples_arr[1].values = {3.0, 0.3, 11.0};
    examples_arr[1].label = 1;
    examples_arr[1].weight = 0.2;
    examples_arr[2].values = {5.0, 0.4, 11.0};
   	examples_arr[2].label = 1;
    examples_arr[2].weight = 0.2;
    examples_arr[3].values = {2.0,  0.2, 22.0};
    examples_arr[3].label = -1;
    examples_arr[3].weight = 0.2;
   	examples_arr[4].values = {4.0, 0.5, 11.0};
    examples_arr[4].label = -1;
    examples_arr[4].weight = 0.2;
    examples.assign(examples_arr, examples_arr + 5);

	AddTreeToModel(examples,&model);
	AddTreeToModel(examples,&model);
	AddTreeToModel(examples,&model);
	
	cout<<SoftClassifyExample(examples[0], model)<<endl;
	cout<<SoftClassifyExample(examples[1], model)<<endl;
	cout<<SoftClassifyExample(examples[2], model)<<endl;
	cout<<SoftClassifyExample(examples[3], model)<<endl;
	cout<<SoftClassifyExample(examples[4], model)<<endl;
	cout<<endl;
	
	float trainX[]={1.0, 0.1, 11.0,3.0, 0.3, 11.0,5.0, 0.4, 11.0,2.0, 0.2, 22.0,4.0, 0.5, 11.0};
	float trainY[]={1,1,1,-1,-1};
	float outputY[]={0,0,0,0,0};
	Model* model_test;
	train(5,3,trainX,trainY,beta,lambda,tree_depth,0,3,&model_test);
	cout<<(*model_test).size()<<endl;
	cout<<(*model_test)[0].first<<endl;
	classify(5,3,trainX,outputY,&model_test);
	cout<<endl;
	cout<<outputY[0]<<endl;
	cout<<outputY[1]<<endl;
	cout<<outputY[2]<<endl;
	cout<<outputY[3]<<endl;
	cout<<outputY[4]<<endl;
}*/

