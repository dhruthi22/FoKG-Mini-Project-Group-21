# Knowledge Graph Fact Checking Project

This project implements a machine learning pipeline for Knowledge Graph (KG) Fact Checking, designed to predict the veracity (truth value) of RDF triple claims. The pipeline processes N-Triples data, extracts global semantic and structural features from entities and predicates, and uses a Gradient Boosting Classifier to score fact veracity for submission to the GERBIL benchmark evaluation platform.

---

## 📌 Features & Pipeline Overview

1. **RDF Parsing**: Parses `.nt` training and test files into structured data using `rdflib`.
2. **Global Feature Engineering**: Computes structural frequency and conditional probability metrics across both train and test sets simultaneously to handle unseen test entities seamlessly:
   * Entity frequencies (`subject`, `object`, `predicate`)
   * Predicate-Object conditional probabilities ($P(\text{object} \mid \text{predicate})$)
   * Object numeric/heuristic checks
3. **Machine Learning Model**: Trains a `GradientBoostingClassifier` to output continuous veracity scores in the range $[0.0, 1.0]$.
4. **GERBIL Serialization**: Generates a strictly formatted `result.ttl` file containing triple veracity predictions formatted as `xsd:double`.

---

## 🛠️ Requirements & Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install rdflib pandas numpy scikit-learn