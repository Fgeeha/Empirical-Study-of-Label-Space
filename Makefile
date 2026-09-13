# =====================================================
# Project: plant-disease-eccv2026
# Makefile — воспроизводимый исследовательский пайплайн
# =====================================================

.DEFAULT_GOAL := help

PYTHON := poetry run python
SCRIPTS := scripts

# Стратегии выравнивания классов
STRATEGIES := fuzzy hybrid sbert

.PHONY: help collect normalize mapping-fuzzy mapping-sbert mapping-hybrid \
        filter split train eval experiment clean

help: ## Показать список целей
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# =====================================================
# Основной пайплайн
# =====================================================

collect: ## Шаг 1: собрать сырые имена классов
	$(PYTHON) $(SCRIPTS)/1.1_collect_classes_pv.py
	$(PYTHON) $(SCRIPTS)/1.2_collect_classes_plantdoc.py
	$(PYTHON) $(SCRIPTS)/1.3_collect_classes_fcdd.py

normalize: ## Шаг 2: нормализовать имена классов (уровень строк)
	$(PYTHON) $(SCRIPTS)/2.2_normalize_classes.py artifacts/classes_pv.txt artifacts/classes_pv_normalized.csv
	$(PYTHON) $(SCRIPTS)/2.2_normalize_classes.py artifacts/classes_target.txt artifacts/classes_target_normalized.csv
	$(PYTHON) $(SCRIPTS)/2.2_normalize_classes.py artifacts/classes_fcdd.txt artifacts/classes_fcdd_normalized.csv

mapping-fuzzy: ## Шаг 3: построить fuzzy-маппинг общего подмножества (baseline)
	$(PYTHON) $(SCRIPTS)/2.4_build_shared_subset.py

mapping-sbert: ## Шаг 4: построить SBERT-маппинг общего подмножества (ablation)
	$(PYTHON) $(SCRIPTS)/2.5_build_shared_subset_sbert.py

mapping-hybrid: ## Шаг 5: построить hybrid-маппинг общего подмножества (финальный)
	$(PYTHON) $(SCRIPTS)/2.6_build_shared_subset_hybrid.py

filter: ## Шаг 6: отфильтровать датасеты (по всем стратегиям)
	$(PYTHON) $(SCRIPTS)/3.1_filter_shared_subset.py --mapping configs/class_mapping_Fuzzy.json
	$(PYTHON) $(SCRIPTS)/3.1_filter_shared_subset.py --mapping configs/class_mapping_hybrid.json
	$(PYTHON) $(SCRIPTS)/3.1_filter_shared_subset.py --mapping configs/class_mapping_sbert.json

split: ## Шаг 7: разбить PlantVillage на train/val/test (по стратегиям)
	for s in $(STRATEGIES); do \
		$(PYTHON) $(SCRIPTS)/4.1_split_pv_dataset.py --strategy $$s ; \
	done

train: ## Шаг 8: обучить ResNet-50 (по стратегиям)
	for s in $(STRATEGIES); do \
		$(PYTHON) $(SCRIPTS)/4.2_train_resnet50_pv.py --strategy $$s ; \
	done

eval: ## Шаг 9: оценить модели на PV test (по стратегиям)
	for s in $(STRATEGIES); do \
		$(PYTHON) $(SCRIPTS)/4.3_eval_resnet50_pv.py --strategy $$s ; \
	done

experiment: collect normalize mapping-fuzzy mapping-hybrid filter split train eval ## Полный эксперимент (рекомендуемый порядок)

# =====================================================
# Обслуживание
# =====================================================

clean: ## Удалить сгенерированные артефакты (безопасно)
	rm -rf data/shared_subset
	rm -rf splits
	rm -rf models/*.pt
	rm -rf results/*.json results/*.csv
	rm -rf figures/*.png
	@echo "[OK] Cleaned generated artifacts"
