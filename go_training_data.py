import argparse
import csv
import datetime
import math
import os
import pickle
import numpy as np
from typing import Dict
from ontobio.ontol_factory import OntologyFactory
from ontobio.util.go_utils import GoAspector


parser = argparse.ArgumentParser(description='Convert GOHierarchy to TALE pickle format')
parser.add_argument('input_file', type=str, help='Input GOWithHierarchy-formatted file for train data')
parser.add_argument('fasta_file', type=str, help='FASTA file to source sequence')
parser.add_argument('ontology_file', type=str, help='GO ontology OBO file')
parser.add_argument('aspect', type=str, default='MF', help='Aspect of annotated terms to output, e.g., MF, BP, CC')
parser.add_argument('pickle_out_dir', type=str, help='Output pickle directory')
parser.add_argument('-m', '--id_forward_mapping', type=str, help='ID forward mapping file, e.g., librarySeqMap')
parser.add_argument('-t', '--test_data_file', type=str, help='Input GOWithHierarchy-formatted file for test data')


EXP_CODES = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"]
EXCLUDED_TERMS = ["GO:0005488", "GO:0005515", "GO:0003674", "GO:0008150", "GO:0005575"]
ONT_ASPECT_NAMESPACES = {"MF": "F", "BP": "P", "CC": "C"}
ONT_ASPECT_TYPES = {"F": "m", "P": "p", "C": "c"}


def parse_fasta_file(fasta_file: str) -> Dict[str, str]:
    fasta_seq_lkp = {}
    seq_id = ''
    with open(fasta_file, 'r') as fasta_f:
        for l in fasta_f:
            if l.startswith('>'):
                seq_id = l.split('>', maxsplit=1)[1].strip()
                fasta_seq_lkp[seq_id] = ''
            else:
                fasta_seq_lkp[seq_id] += l.strip()
    return fasta_seq_lkp


def parse_id_forward_mapping(id_forward_mapping_file: str) -> Dict[str, str]:
    id_fwd_lkp = {}
    with open(id_forward_mapping_file, 'r') as id_fwd_f:
        reader = csv.reader(id_fwd_f, delimiter='\t')
        for row in reader:
            prev_id = row[0]
            new_id = row[1]
            id_fwd_lkp[prev_id] = new_id
    return id_fwd_lkp


def parse_ontology_aspects(ont_file: str) -> Dict[str, str]:
    aspect_lkp = {}
    with open(ont_file, 'r') as ont_f:
        term = None
        namespace = None
        children = {}
        for l in ont_f.readlines():
            if l.startswith('[Term]'):
                term = None
                namespace = None
            elif l.startswith('id: '):
                term = l.split(' ', maxsplit=1)[1].strip()
            elif l.startswith('namespace: '):
                namespace = l.split(' ', maxsplit=1)[1].strip()
            if term and namespace:
                aspect_lkp[term] = namespace
    return aspect_lkp


class TermIndManager:
    def __init__(self):
        self.term_ind_lkp = {}
        self.term_ind_counter = 0
        self.label_regular = set()  # parent, child indices of is_a or part_of related terms
        self.label_matrix = {
            'GO:0008150': set(),
            'GO:0003674': set(),
            'GO:0005575': set()
        }  # indices of is_a or part_of related ancestor terms

    def get_ind(self, ind_term):
        if ind_term not in self.term_ind_lkp:
            self.term_ind_lkp[ind_term] = self.term_ind_counter
            self.term_ind_counter += 1
        return self.term_ind_lkp[ind_term]

    def add_label_to_matrix(self, term, label):
        if term not in self.label_matrix:
            self.label_matrix[term] = set()
        self.label_matrix[term].add(label)


class GoTrainingData:
    def __init__(self, go_ontology, go_aspector: GoAspector, fasta_sequence_lkp: Dict = None):
        self.term_ind_manager = TermIndManager()
        self.go_ontology = go_ontology
        self.go_aspector = go_aspector
        self.fasta_sequence_lkp = fasta_sequence_lkp
        self.child_cache = {}
        self.parent_cache = {}
        self.ancestor_cache = {}

    def get_annots_and_labels(self, genes, mode: str = 'train'):
        annots = []
        labels = []
        for seq_id, go_terms in genes.items():
            annot_labels = set()
            for gt in go_terms:
                gt_ind= self.term_ind_manager.get_ind(gt)
                annot_labels.add(gt_ind)
                for anc_term in self.get_ancestors(gt):
                    anc_term_ind = self.term_ind_manager.get_ind(anc_term)
                    annot_labels.add(anc_term_ind)
                    self.term_ind_manager.add_label_to_matrix(gt, anc_term_ind)
                # Register children in term_ind_manager
                for c in self.get_children(gt):
                    parent_child_ind = [gt_ind, self.term_ind_manager.get_ind(c)]
                    self.term_ind_manager.label_regular.add(tuple(parent_child_ind))
                for p in self.get_parents(gt):
                    parent_child_ind = [self.term_ind_manager.get_ind(p), gt_ind]
                    self.term_ind_manager.label_regular.add(tuple(parent_child_ind))
            uniprot_id = seq_id.split('|')[-1].split('=')[-1]
            # annot_date = datetime.datetime.strptime(date, '%Y%m%d').strftime('%d-%b-%Y').upper()
            annot_date = '08-MAY-2019'  # QfO Ref Prot release date for 2019_04
            seq = self.fasta_sequence_lkp.get(seq_id)
            if seq is None or len(seq) > 1000:
                continue

            annot = {
                'ac': uniprot_id,
                'date': annot_date,
                'seq': seq,
                'mode': mode,
                'GO': list(go_terms),
                'label': list(annot_labels)  # indices of is_a ancestors for all go_terms
            }
            annots.append(annot)
            labels.append(list(annot_labels))
        return annots, labels

    def filter_out_differing_aspect_terms(self, orig_term, go_terms):
        # orig_term is the term to compare aspect with
        return [gt for gt in go_terms if self.go_aspector.go_aspect(gt) == self.go_aspector.go_aspect(orig_term)]

    def get_children(self, term):
        if term not in self.child_cache:
            child_ont = self.go_ontology.subontology(self.go_ontology.descendants(term, reflexive=True))
            children = child_ont.children(term, relations=['subClassOf', 'BFO:0000050'])
            aspect_children = self.filter_out_differing_aspect_terms(term, children)
            self.child_cache[term] = aspect_children
        return self.child_cache[term]

    def get_parents(self, term):
        if term not in self.parent_cache:
            parent_ont = self.go_ontology.subontology(self.go_ontology.ancestors(term, reflexive=True))
            parents = parent_ont.parents(term, relations=['subClassOf', 'BFO:0000050'])
            aspect_parents = self.filter_out_differing_aspect_terms(term, parents)
            self.parent_cache[term] = aspect_parents
        return self.parent_cache[term]

    def get_ancestors(self, term):
        if term not in self.ancestor_cache:
            anc_ont = self.go_ontology.subontology(self.go_ontology.ancestors(term, reflexive=True))
            ancestors = anc_ont.ancestors(term, relations=['subClassOf', 'BFO:0000050'])
            aspect_ancestors = self.filter_out_differing_aspect_terms(term, ancestors)
            self.ancestor_cache[term] = aspect_ancestors
        return self.ancestor_cache[term]


def parse_go_hierarchy_file(go_hier_file: str, asp: str, go_data_manager: GoTrainingData, id_fwd_map: Dict = None) -> Dict[str, str]:
    genes = {}
    with open(go_hier_file, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in reader:
            sequence_id = row['SequenceID']
            go_term = row['GOHierarchy'].split('>')[0]
            evidence_code = row['EvidenceCode']
            with_info = row['With'] if row['With'] else ''
            reference = row['Reference']
            date = row['Date']  # 20190727 # The date of the sequence released in Swiss-Prot
            db = row['DB']

            if evidence_code not in EXP_CODES:
                continue

            if go_term in EXCLUDED_TERMS:
                continue

            if go_term not in go_data_manager.go_ontology.nodes():
                continue

            if go_data_manager.go_aspector.go_aspect(go_term) != ONT_ASPECT_NAMESPACES[asp]:
                continue

            if id_fwd_map:
                sequence_id = id_fwd_map.get(sequence_id)
                if sequence_id is None:
                    # Skip if sequence_id is not found in the id_forward_mapping
                    continue

            if sequence_id not in genes:
                genes[sequence_id] = set()
            genes[sequence_id].add(go_term)
    return genes


if __name__ == "__main__":
    args = parser.parse_args()

    fasta_sequence_lkp = parse_fasta_file(args.fasta_file)
    # term_aspects = parse_ontology_aspects(args.ontology_file)
    go_ont = OntologyFactory().create(args.ontology_file)
    aspector = GoAspector(go_ont)
    data_mgr = GoTrainingData(go_ont, aspector, fasta_sequence_lkp)
    id_forward_mapping = None
    if args.id_forward_mapping:
        id_forward_mapping = parse_id_forward_mapping(args.id_forward_mapping)

    training_genes = parse_go_hierarchy_file(args.input_file, args.aspect, data_mgr, id_forward_mapping)
    test_genes = None
    if args.test_data_file:
        test_genes = parse_go_hierarchy_file(args.test_data_file, args.aspect, data_mgr)

    train_annots, train_labels = data_mgr.get_annots_and_labels(training_genes)
    print("{} training sequences".format(len(train_annots)))
    test_annots, test_labels = None, None
    if test_genes:
        # Extracting test annots and labels adds their unique terms to the term_ind_manager
        test_annots, test_labels = data_mgr.get_annots_and_labels(test_genes, mode='test')
        print("{} test sequences".format(len(test_annots)))
        # filter out any test_annots if sequence is already in train_annots and remove by index from test_labels
        train_seqs = set([a['ac'] for a in train_annots])
        for i, annot in enumerate(test_annots):
            if annot['ac'] in train_seqs:
                test_annots.pop(i)
                test_labels.pop(i)
        print("{} test sequences not in training set".format(len(test_annots)))
        max_test_count = math.ceil(len(train_annots) * 0.05)  # 5% of training set
        test_annots = test_annots[:max_test_count]
        test_labels = test_labels[:max_test_count]
        print("{} test sequences in final test set".format(len(test_annots)))

    terms = {}
    # ind_counter = 0
    for term, t_ind in data_mgr.term_ind_manager.term_ind_lkp.items():
        go_term_ds = {
            'father': data_mgr.get_parents(term),
            'child': data_mgr.get_children(term),
            'name': go_ont.label(term),
            'type': ONT_ASPECT_TYPES[data_mgr.go_aspector.go_aspect(term)],  # 'm' for MF, 'p' for BP, 'c' for CC
            'ind': t_ind
        }
        # ind_counter += 1
        terms[term] = go_term_ds
    print("{} GO terms".format(len(terms)))

    # Construct the confusing "label matrix" file
    label_matrix_1_sparse = []
    # First, iterate over terms filling in empty and getting max length
    max_length = 0
    for t in terms:
        if t not in data_mgr.term_ind_manager.label_matrix:
            for anc_term in data_mgr.get_ancestors(t):
                anc_ind = data_mgr.term_ind_manager.get_ind(anc_term)
                data_mgr.term_ind_manager.add_label_to_matrix(t, anc_ind)
        term_label_matrix = list(data_mgr.term_ind_manager.label_matrix[t])
        max_length = max(max_length, len(term_label_matrix))
    print("Max length of label matrix: {}".format(max_length))
    num_terms = len(terms)
    # Iterate again to normalize every term's label matrix to max length and padding with value of num_terms
    for t in terms:
        term_label_matrix = list(data_mgr.term_ind_manager.label_matrix[t])
        term_label_matrix.extend([num_terms] * (max_length - len(term_label_matrix)))
        label_matrix_1_sparse.append(term_label_matrix)

    # Write to pickle files
    seq_out_path = os.path.join(args.pickle_out_dir, "train_seq_{}".format(args.aspect.lower()))
    with open(seq_out_path, 'wb') as f:
        pickle.dump(train_annots, f)

    label_out_path = os.path.join(args.pickle_out_dir, "train_label_{}".format(args.aspect.lower()))
    with open(label_out_path, 'wb') as f:
        pickle.dump(train_labels, f)

    ont_out_path = os.path.join(args.pickle_out_dir, "{}_go_1.pickle".format(args.aspect.lower()))
    with open(ont_out_path, 'wb') as f:
        pickle.dump(terms, f)

    label_regular_1 = np.array(([list(l) for l in data_mgr.term_ind_manager.label_regular]))
    label_regular_1_path = os.path.join(args.pickle_out_dir, "{}_label_regular_1".format(args.aspect.lower()))
    np.save(label_regular_1_path, label_regular_1)

    label_matrix_1_sparse_array = np.array(label_matrix_1_sparse)
    label_matrix_1_sparse_path = os.path.join(args.pickle_out_dir, "{}_label_matrix_1_sparse".format(args.aspect.lower()))
    np.save(label_matrix_1_sparse_path, label_matrix_1_sparse_array)

    if test_annots and test_labels:
        seq_out_path = os.path.join(args.pickle_out_dir, "test_seq_{}".format(args.aspect.lower()))
        with open(seq_out_path, 'wb') as f:
            pickle.dump(test_annots, f)

        label_out_path = os.path.join(args.pickle_out_dir, "test_label_{}".format(args.aspect.lower()))
        with open(label_out_path, 'wb') as f:
            pickle.dump(test_labels, f)

