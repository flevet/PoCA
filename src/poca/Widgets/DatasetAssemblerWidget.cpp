/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DatasetAssemblerWidget.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*
* PoCA is a free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* The algorithms that underlie PoCA have required considerable
* development. They are described in the original SR-Tesseler paper,
* doi:10.1038/nmeth.3579. If you use PoCA as part of work (visualization,
* manipulation, quantification) towards a scientific publication, please include
* a citation to the original paper.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <algorithm>
#include <fstream>

#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QVBoxLayout>
#include <QtCore/QDir>
#include <QtCore/QDirIterator>
#include <QtCore/QFileInfo>
#include <QtCore/QRegularExpression>
#include <QtCore/QSet>
#include <QtCore/QSignalBlocker>
#include <QtGui/QColor>

#include <General/Engine.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Objects/MyMultipleObject.hpp>

#include "../Widgets/DatasetAssemblerWidget.hpp"

namespace {
	QString buildDatasetName(const QString& rootFolder, const QString& key, const bool prefixRootName, const QString& separator)
	{
		const QString rootName = QFileInfo(rootFolder).fileName();
		if (prefixRootName && !rootName.isEmpty() && !key.isEmpty())
			return rootName + separator + key;
		if (prefixRootName && !rootName.isEmpty())
			return rootName;
		if (!key.isEmpty())
			return key;
		return rootName;
	}

	QTableWidgetItem* createTextItem(const QString& text = QString())
	{
		auto* item = new QTableWidgetItem(text);
		item->setFlags(item->flags() | Qt::ItemIsEditable);
		return item;
	}

	QTableWidgetItem* createCheckItem(const bool checked)
	{
		auto* item = new QTableWidgetItem;
		item->setFlags((item->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled) & ~Qt::ItemIsEditable);
		item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
		return item;
	}
}

DatasetAssemblerWidget::DatasetAssemblerWidget(QWidget* _parent)
	: QWidget(_parent)
{
	setObjectName("DatasetAssemblerWidget");

	m_rootsList = new QListWidget(this);
	m_rootsList->setSelectionMode(QAbstractItemView::ExtendedSelection);

	m_addFolderButton = new QPushButton("Add folder(s)", this);
	m_removeFolderButton = new QPushButton("Remove selected", this);
	connect(m_addFolderButton, SIGNAL(released()), this, SLOT(onAddFolder()));
	connect(m_removeFolderButton, SIGNAL(released()), this, SLOT(onRemoveFolder()));

	QHBoxLayout* rootsButtonsLayout = new QHBoxLayout;
	rootsButtonsLayout->addWidget(m_addFolderButton);
	rootsButtonsLayout->addWidget(m_removeFolderButton);
	rootsButtonsLayout->addStretch(1);

	QGroupBox* rootsGroup = new QGroupBox("Root folders", this);
	QVBoxLayout* rootsLayout = new QVBoxLayout;
	rootsLayout->addLayout(rootsButtonsLayout);
	rootsLayout->addWidget(m_rootsList);
	rootsGroup->setLayout(rootsLayout);

	m_rulesTable = new QTableWidget(this);
	m_rulesTable->setColumnCount(6);
	QStringList headers;
	headers << "On" << "Req" << "Label" << "Relative folder" << "Filename regex" << "Key group";
	m_rulesTable->setHorizontalHeaderLabels(headers);
	m_rulesTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
	m_rulesTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
	m_rulesTable->verticalHeader()->setVisible(false);
	m_rulesTable->setSelectionBehavior(QAbstractItemView::SelectRows);
	m_rulesTable->setSelectionMode(QAbstractItemView::SingleSelection);

	m_addRuleButton = new QPushButton("Add rule", this);
	m_removeRuleButton = new QPushButton("Remove rule", this);
	m_importJsonButton = new QPushButton("Import JSON", this);
	m_exportJsonButton = new QPushButton("Export JSON", this);
	connect(m_addRuleButton, SIGNAL(released()), this, SLOT(onAddRule()));
	connect(m_removeRuleButton, SIGNAL(released()), this, SLOT(onRemoveRule()));
	connect(m_importJsonButton, SIGNAL(released()), this, SLOT(onImportJson()));
	connect(m_exportJsonButton, SIGNAL(released()), this, SLOT(onExportJson()));
	connect(m_rulesTable, SIGNAL(itemChanged(QTableWidgetItem*)), this, SLOT(onRulesChanged()));

	QHBoxLayout* rulesButtonsLayout = new QHBoxLayout;
	rulesButtonsLayout->addWidget(m_addRuleButton);
	rulesButtonsLayout->addWidget(m_removeRuleButton);
	rulesButtonsLayout->addWidget(m_importJsonButton);
	rulesButtonsLayout->addWidget(m_exportJsonButton);
	rulesButtonsLayout->addStretch(1);

	QGroupBox* namingGroup = new QGroupBox("Naming", this);
	m_prefixRootNameCBox = new QCheckBox("Prefix dataset names with root folder name", this);
	m_prefixRootNameCBox->setChecked(true);
	QLabel* separatorLabel = new QLabel("Separator", this);
	m_nameSeparatorEdit = new QLineEdit("_", this);
	m_nameSeparatorEdit->setMaximumWidth(60);
	QHBoxLayout* namingLayout = new QHBoxLayout;
	namingLayout->addWidget(m_prefixRootNameCBox);
	namingLayout->addStretch(1);
	namingLayout->addWidget(separatorLabel);
	namingLayout->addWidget(m_nameSeparatorEdit);
	namingGroup->setLayout(namingLayout);

	QGroupBox* rulesGroup = new QGroupBox("Component rules", this);
	QVBoxLayout* rulesLayout = new QVBoxLayout;
	rulesLayout->addLayout(rulesButtonsLayout);
	rulesLayout->addWidget(m_rulesTable);
	rulesLayout->addWidget(namingGroup);
	rulesGroup->setLayout(rulesLayout);

	m_previewTree = new QTreeWidget(this);
	m_previewTree->setColumnCount(2);
	m_previewTree->setHeaderLabels(QStringList() << "Hierarchy" << "Type");
	m_previewTree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
	m_previewTree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
	m_previewTree->setMinimumHeight(180);

	m_logEdit = new QTextEdit(this);
	m_logEdit->setReadOnly(true);
	m_logEdit->setMinimumHeight(180);

	m_previewButton = new QPushButton("Preview", this);
	m_assembleButton = new QPushButton("Assemble datasets", this);
	connect(m_previewButton, SIGNAL(released()), this, SLOT(onPreview()));
	connect(m_assembleButton, SIGNAL(released()), this, SLOT(onAssemble()));

	QHBoxLayout* actionsLayout = new QHBoxLayout;
	actionsLayout->addStretch(1);
	actionsLayout->addWidget(m_previewButton);
	actionsLayout->addWidget(m_assembleButton);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->addWidget(rootsGroup);
	mainLayout->addWidget(rulesGroup);
	mainLayout->addWidget(new QLabel("Hierarchy preview", this));
	mainLayout->addWidget(m_previewTree);
	mainLayout->addWidget(new QLabel("Log", this));
	mainLayout->addWidget(m_logEdit);
	mainLayout->addLayout(actionsLayout);
	setLayout(mainLayout);

	setRulesToTable({ defaultRule() });
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::loadParameters(const nlohmann::json& _json)
{
	const std::string nameStr = objectName().toStdString();
	if (!_json.contains(nameStr))
		return;

	const nlohmann::json& json = _json[nameStr];

	m_rootsList->clear();
	if (json.contains("rootFolders")) {
		try {
			std::vector<std::string> roots = json["rootFolders"].get<std::vector<std::string>>();
			for (const std::string& root : roots)
				m_rootsList->addItem(root.c_str());
		}
		catch (nlohmann::json::exception&) {}
	}

	if (json.contains("lastRootPath")) {
		try { m_lastRootPath = json["lastRootPath"].get<std::string>().c_str(); }
		catch (nlohmann::json::exception&) {}
	}
	if (json.contains("prefixRootName")) {
		try { m_prefixRootNameCBox->setChecked(json["prefixRootName"].get<bool>()); }
		catch (nlohmann::json::exception&) {}
	}
	if (json.contains("nameSeparator")) {
		try { m_nameSeparatorEdit->setText(json["nameSeparator"].get<std::string>().c_str()); }
		catch (nlohmann::json::exception&) {}
	}
	if (json.contains("rules")) {
		std::vector<DatasetRule> rules;
		try {
			for (const auto& jsonRule : json["rules"]) {
				DatasetRule rule;
				if (jsonRule.contains("enabled")) rule.enabled = jsonRule["enabled"].get<bool>();
				if (jsonRule.contains("required")) rule.required = jsonRule["required"].get<bool>();
				if (jsonRule.contains("label")) rule.label = jsonRule["label"].get<std::string>().c_str();
				if (jsonRule.contains("relativeFolder")) rule.relativeFolder = jsonRule["relativeFolder"].get<std::string>().c_str();
				if (jsonRule.contains("regex")) rule.regex = jsonRule["regex"].get<std::string>().c_str();
				if (jsonRule.contains("keyCaptureGroup")) rule.keyCaptureGroup = jsonRule["keyCaptureGroup"].get<int>();
				rules.push_back(rule);
			}
		}
		catch (nlohmann::json::exception&) {}
		if (!rules.empty())
			setRulesToTable(rules);
	}
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::saveParameters(nlohmann::json& _json) const
{
	const std::string nameStr = objectName().toStdString();
	nlohmann::json& json = _json[nameStr];
	json["lastRootPath"] = m_lastRootPath.toStdString();
	json["prefixRootName"] = m_prefixRootNameCBox->isChecked();
	json["nameSeparator"] = m_nameSeparatorEdit->text().toStdString();

	std::vector<std::string> roots;
	for (int row = 0; row < m_rootsList->count(); ++row)
		roots.push_back(m_rootsList->item(row)->text().toStdString());
	json["rootFolders"] = roots;

	std::vector<nlohmann::json> rulesJson;
	for (const DatasetRule& rule : rulesFromTable()) {
		nlohmann::json jsonRule;
		jsonRule["enabled"] = rule.enabled;
		jsonRule["required"] = rule.required;
		jsonRule["label"] = rule.label.toStdString();
		jsonRule["relativeFolder"] = rule.relativeFolder.toStdString();
		jsonRule["regex"] = rule.regex.toStdString();
		jsonRule["keyCaptureGroup"] = rule.keyCaptureGroup;
		rulesJson.push_back(jsonRule);
	}
	json["rules"] = rulesJson;
}

void DatasetAssemblerWidget::onAddFolder()
{
	QString startPath = m_lastRootPath.isEmpty() ? QDir::currentPath() : m_lastRootPath;
	const QString folder = QFileDialog::getExistingDirectory(this, tr("Select root folder"), startPath, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
	if (folder.isEmpty())
		return;

	m_lastRootPath = folder;
	const QString absFolder = QFileInfo(folder).absoluteFilePath();
	for (int row = 0; row < m_rootsList->count(); ++row)
		if (m_rootsList->item(row)->text() == absFolder)
			return;
	m_rootsList->addItem(absFolder);
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onRemoveFolder()
{
	qDeleteAll(m_rootsList->selectedItems());
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onAddRule()
{
	const int row = m_rulesTable->rowCount();
	m_rulesTable->insertRow(row);

	const DatasetRule rule = defaultRule();
	m_rulesTable->setItem(row, 0, createCheckItem(rule.enabled));
	m_rulesTable->setItem(row, 1, createCheckItem(rule.required));
	m_rulesTable->setItem(row, 2, createTextItem(rule.label));
	m_rulesTable->setItem(row, 3, createTextItem(rule.relativeFolder));
	m_rulesTable->setItem(row, 4, createTextItem(rule.regex));
	m_rulesTable->setItem(row, 5, createTextItem(QString::number(rule.keyCaptureGroup)));
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onRemoveRule()
{
	const int row = m_rulesTable->currentRow();
	if (row >= 0)
		m_rulesTable->removeRow(row);
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onImportJson()
{
	const QString startPath = m_lastRootPath.isEmpty() ? QDir::currentPath() : m_lastRootPath;
	const QString filename = QFileDialog::getOpenFileName(this, tr("Import assembler settings"), startPath, tr("JSON files (*.json)"));
	if (filename.isEmpty())
		return;

	try {
		std::ifstream fs(filename.toStdString());
		if (!fs)
			throw std::runtime_error("Could not open file");
		nlohmann::json json;
		fs >> json;
		loadParameters(json);
		appendLog(QString("Imported assembler settings from %1").arg(filename));
	}
	catch (const std::exception& e) {
		QMessageBox::warning(this, tr("Assembler"), tr("Failed to import JSON: %1").arg(e.what()));
	}
}

void DatasetAssemblerWidget::onExportJson()
{
	const QString startPath = m_lastRootPath.isEmpty() ? QDir::currentPath() : m_lastRootPath;
	const QString filename = QFileDialog::getSaveFileName(this, tr("Export assembler settings"), startPath + "/dataset_assembler.json", tr("JSON files (*.json)"));
	if (filename.isEmpty())
		return;

	try {
		nlohmann::json json;
		saveParameters(json);
		std::ofstream fs(filename.toStdString());
		if (!fs)
			throw std::runtime_error("Could not open file");
		fs << json.dump(2);
		appendLog(QString("Exported assembler settings to %1").arg(filename));
	}
	catch (const std::exception& e) {
		QMessageBox::warning(this, tr("Assembler"), tr("Failed to export JSON: %1").arg(e.what()));
	}
}

void DatasetAssemblerWidget::onPreview()
{
	m_logEdit->clear();

	const std::vector<DatasetRule> rules = rulesFromTable();
	QStringList roots;
	for (int row = 0; row < m_rootsList->count(); ++row)
		roots << m_rootsList->item(row)->text();

	QStringList errors, warnings;
	if (!validateConfiguration(rules, roots, errors, warnings)) {
		for (const QString& error : errors)
			appendLog(QString("Error: %1").arg(error));
		for (const QString& warning : warnings)
			appendLog(QString("Warning: %1").arg(warning));
		QMessageBox::warning(this, tr("Assembler"), tr("The assembler configuration is invalid. See the log for details."));
		return;
	}

	populatePreviewTree(roots, rules);

	const bool prefixRootName = m_prefixRootNameCBox->isChecked();
	const QString separator = m_nameSeparatorEdit->text().isEmpty() ? "_" : m_nameSeparatorEdit->text();
	int validDatasetCount = 0, skippedDatasetCount = 0;

	for (const QString& rootFolder : roots) {
		appendLog(QString("Scanning root folder: %1").arg(rootFolder));
		const ScanResult scan = scanRootFolder(rootFolder, rules);
		for (const QString& message : scan.messages)
			appendLog(message);
		appendLog(QString("Matched %1 file(s) across %2 dataset key(s).").arg(scan.matchedFiles).arg(scan.datasets.size()));

		for (auto it = scan.datasets.begin(); it != scan.datasets.end(); ++it) {
			const QString datasetKey = it.key();
			const DatasetEntry& entry = it.value();

			bool missingRequired = false;
			QStringList matchedLabels, missingLabels;
			for (int ruleIndex = 0; ruleIndex < (int)rules.size(); ++ruleIndex) {
				const DatasetRule& rule = rules[ruleIndex];
				if (!rule.enabled)
					continue;
				if (entry.filesByRule.contains(ruleIndex))
					matchedLabels << ruleDisplayName(rule, ruleIndex);
				else if (rule.required) {
					missingRequired = true;
					missingLabels << ruleDisplayName(rule, ruleIndex);
				}
			}

			const QString objectName = buildDatasetName(rootFolder, datasetKey, prefixRootName, separator);
			if (missingRequired || entry.filesByRule.isEmpty()) {
				appendLog(QString("Skip [%1] -> %2 | matched: %3 | missing required: %4")
					.arg(datasetKey)
					.arg(objectName.isEmpty() ? datasetKey : objectName)
					.arg(matchedLabels.join(", "))
					.arg(missingLabels.join(", ")));
				++skippedDatasetCount;
				continue;
			}

			appendLog(QString("Ready [%1] -> %2 | %3 component(s): %4")
				.arg(datasetKey)
				.arg(objectName.isEmpty() ? datasetKey : objectName)
				.arg(entry.filesByRule.size())
				.arg(matchedLabels.join(", ")));
			++validDatasetCount;
		}
	}

	appendLog(QString("Preview summary: %1 dataset(s) ready, %2 skipped.").arg(validDatasetCount).arg(skippedDatasetCount));
}

void DatasetAssemblerWidget::onAssemble()
{
	m_logEdit->clear();

	const std::vector<DatasetRule> rules = rulesFromTable();
	QStringList roots;
	for (int row = 0; row < m_rootsList->count(); ++row)
		roots << m_rootsList->item(row)->text();

	QStringList errors, warnings;
	if (!validateConfiguration(rules, roots, errors, warnings)) {
		for (const QString& error : errors)
			appendLog(QString("Error: %1").arg(error));
		for (const QString& warning : warnings)
			appendLog(QString("Warning: %1").arg(warning));
		QMessageBox::warning(this, tr("Assembler"), tr("The assembler configuration is invalid. See the log for details."));
		return;
	}

	populatePreviewTree(roots, rules);

	poca::core::Engine* engine = poca::core::Engine::instance();
	std::vector<poca::core::MyObjectInterface*> objects;
	std::vector<AssembledDatasetInfo> assembledInfos;
	const bool prefixRootName = m_prefixRootNameCBox->isChecked();
	const QString separator = m_nameSeparatorEdit->text().isEmpty() ? "_" : m_nameSeparatorEdit->text();

	for (const QString& rootFolder : roots) {
		appendLog(QString("Scanning root folder: %1").arg(rootFolder));
		const ScanResult scan = scanRootFolder(rootFolder, rules);
		for (const QString& message : scan.messages)
			appendLog(message);

		for (auto it = scan.datasets.begin(); it != scan.datasets.end(); ++it) {
			const QString datasetKey = it.key();
			const DatasetEntry& entry = it.value();

			bool missingRequired = false;
			for (int ruleIndex = 0; ruleIndex < (int)rules.size(); ++ruleIndex) {
				const DatasetRule& rule = rules[ruleIndex];
				if (!rule.enabled || !rule.required)
					continue;
				if (!entry.filesByRule.contains(ruleIndex)) {
					missingRequired = true;
					appendLog(QString("Dataset [%1] skipped, missing required rule [%2].").arg(datasetKey).arg(ruleDisplayName(rule, ruleIndex)));
				}
			}
			if (missingRequired || entry.filesByRule.isEmpty())
				continue;

			poca::core::CommandInfo firstLoadInfo(false, "open", "path", entry.filesByRule.begin().value().toStdString());
			poca::core::MyObjectInterface* object = engine->loadDataAndCreateObject(entry.filesByRule.begin().value(), &firstLoadInfo);
			if (object == nullptr) {
				appendLog(QString("Failed to create object for dataset [%1] from %2").arg(datasetKey).arg(entry.filesByRule.begin().value()));
				continue;
			}

			bool valid = true;
			auto fileIt = entry.filesByRule.begin();
			++fileIt;
			for (; fileIt != entry.filesByRule.end(); ++fileIt) {
				poca::core::CommandInfo addInfo(false, "open", "path", fileIt.value().toStdString());
				if (!engine->loadDataAndAddToObject(fileIt.value(), object, &addInfo)) {
					appendLog(QString("Failed to add component %1 to dataset [%2]").arg(fileIt.value()).arg(datasetKey));
					valid = false;
					break;
				}
			}

			if (!valid) {
				delete object;
				continue;
			}

			const QString objectName = buildDatasetName(rootFolder, datasetKey, prefixRootName, separator);
			if (!objectName.isEmpty())
				object->setName(objectName.toStdString());
			objects.push_back(object);
			assembledInfos.push_back({ rootFolder, datasetKey, objectName.isEmpty() ? datasetKey : objectName, hierarchySegmentsForDatasetFolder(rootFolder, entry.datasetFolder), object });
			appendLog(QString("Created dataset [%1] with %2 component(s).").arg(objectName.isEmpty() ? datasetKey : objectName).arg(entry.filesByRule.size()));
		}
	}

	if (objects.empty()) {
		QMessageBox::information(this, tr("Assembler"), tr("No dataset could be assembled with the current rules."));
		return;
	}

	poca::core::MyObjectInterface* createdObject = objects.size() == 1 ? objects.front() : engine->generateMultipleObject(objects);
	if (createdObject == nullptr) {
		QMessageBox::warning(this, tr("Assembler"), tr("The datasets were created but the final object could not be assembled."));
		return;
	}

	MyMultipleObject* multipleObject = dynamic_cast<MyMultipleObject*>(createdObject);
	if (multipleObject != nullptr)
		populateHierarchy(multipleObject, assembledInfos);

	appendLog(QString("Created %1 object(s).").arg(objects.size()));
	emit transferNewObjectCreated(createdObject);
}

void DatasetAssemblerWidget::onRulesChanged()
{
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::appendLog(const QString& _text)
{
	m_logEdit->append(_text);
}

std::vector<DatasetAssemblerWidget::DatasetRule> DatasetAssemblerWidget::rulesFromTable() const
{
	std::vector<DatasetRule> rules;
	for (int row = 0; row < m_rulesTable->rowCount(); ++row) {
		DatasetRule rule;
		if (m_rulesTable->item(row, 0) != nullptr)
			rule.enabled = m_rulesTable->item(row, 0)->checkState() == Qt::Checked;
		if (m_rulesTable->item(row, 1) != nullptr)
			rule.required = m_rulesTable->item(row, 1)->checkState() == Qt::Checked;
		if (m_rulesTable->item(row, 2) != nullptr)
			rule.label = m_rulesTable->item(row, 2)->text();
		if (m_rulesTable->item(row, 3) != nullptr)
			rule.relativeFolder = m_rulesTable->item(row, 3)->text();
		if (m_rulesTable->item(row, 4) != nullptr)
			rule.regex = m_rulesTable->item(row, 4)->text();
		if (m_rulesTable->item(row, 5) != nullptr)
			rule.keyCaptureGroup = std::max(0, m_rulesTable->item(row, 5)->text().toInt());
		if (!rule.regex.trimmed().isEmpty())
			rules.push_back(rule);
	}
	return rules;
}

void DatasetAssemblerWidget::setRulesToTable(const std::vector<DatasetRule>& _rules)
{
	m_rulesTable->setRowCount(0);
	for (const DatasetRule& rule : _rules) {
		const int row = m_rulesTable->rowCount();
		m_rulesTable->insertRow(row);
		m_rulesTable->setItem(row, 0, createCheckItem(rule.enabled));
		m_rulesTable->setItem(row, 1, createCheckItem(rule.required));
		m_rulesTable->setItem(row, 2, createTextItem(rule.label));
		m_rulesTable->setItem(row, 3, createTextItem(rule.relativeFolder));
		m_rulesTable->setItem(row, 4, createTextItem(rule.regex));
		m_rulesTable->setItem(row, 5, createTextItem(QString::number(rule.keyCaptureGroup)));
	}
	refreshRulesFeedback();
}

DatasetAssemblerWidget::DatasetRule DatasetAssemblerWidget::defaultRule() const
{
	DatasetRule rule;
	rule.label = "raw";
	rule.regex = "(.*)";
	return rule;
}

bool DatasetAssemblerWidget::validateConfiguration(const std::vector<DatasetRule>& _rules, const QStringList& _roots, QStringList& _errors, QStringList& _warnings) const
{
	if (_rules.empty())
		_errors << "Please define at least one rule with a regex.";
	if (_roots.isEmpty())
		_errors << "Please add at least one root folder.";

	QSet<QString> enabledLabels;
	bool hasEnabledRule = false;
	for (int index = 0; index < (int)_rules.size(); ++index) {
		const DatasetRule& rule = _rules[index];
		if (!rule.enabled)
			continue;

		hasEnabledRule = true;
		const QString displayName = ruleDisplayName(rule, index);
		const QString trimmedRegex = rule.regex.trimmed();
		if (trimmedRegex.isEmpty())
			_errors << QString("Rule [%1] has an empty regex.").arg(displayName);

		const QRegularExpression regex(trimmedRegex);
		if (!regex.isValid())
			_errors << QString("Rule [%1] has an invalid regex: %2").arg(displayName).arg(regex.errorString());
		else if (rule.keyCaptureGroup > regex.captureCount())
			_errors << QString("Rule [%1] requests key capture group %2 but the regex only defines %3 capture group(s).")
				.arg(displayName).arg(rule.keyCaptureGroup).arg(regex.captureCount());

		const QString labelKey = rule.label.trimmed().toLower();
		if (!labelKey.isEmpty()) {
			if (enabledLabels.contains(labelKey))
				_warnings << QString("Rule label [%1] is used more than once.").arg(rule.label.trimmed());
			enabledLabels.insert(labelKey);
		}
	}

	if (!hasEnabledRule)
		_errors << "At least one rule must be enabled.";

	for (const QString& root : _roots) {
		const QFileInfo info(root);
		if (!info.exists() || !info.isDir())
			_errors << QString("Root folder does not exist: %1").arg(root);
	}

	return _errors.isEmpty();
}

DatasetAssemblerWidget::ScanResult DatasetAssemblerWidget::scanRootFolder(const QString& _rootFolder, const std::vector<DatasetRule>& _rules) const
{
	ScanResult result;
	const QStringList datasetFolders = discoverDatasetFolders(_rootFolder, _rules);
	result.messages << QString("Discovered %1 dataset folder(s) under root [%2].").arg(datasetFolders.size()).arg(_rootFolder);

	for (const QString& datasetFolder : datasetFolders) {
		const QString datasetFolderName = QFileInfo(datasetFolder).fileName().isEmpty() ? datasetFolder : QFileInfo(datasetFolder).fileName();
		for (int ruleIndex = 0; ruleIndex < (int)_rules.size(); ++ruleIndex) {
			const DatasetRule& rule = _rules[ruleIndex];
			if (!rule.enabled)
				continue;

			QDir baseDir(datasetFolder);
			if (!rule.relativeFolder.trimmed().isEmpty())
				baseDir = QDir(baseDir.filePath(rule.relativeFolder.trimmed()));

			if (!baseDir.exists()) {
				result.messages << QString("Dataset folder [%1], rule [%2] skipped, folder does not exist: %3")
					.arg(datasetFolderName).arg(ruleDisplayName(rule, ruleIndex)).arg(baseDir.absolutePath());
				continue;
			}

			const QRegularExpression regex(rule.regex);
			if (!regex.isValid()) {
				result.messages << QString("Rule [%1] skipped, invalid regex: %2").arg(ruleDisplayName(rule, ruleIndex)).arg(rule.regex);
				continue;
			}

			int matchedForRule = 0;
			QDirIterator it(baseDir.absolutePath(), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::NoIteratorFlags);
			while (it.hasNext()) {
				const QString absPath = it.next();
				const QFileInfo info(absPath);
				const QRegularExpressionMatch match = regex.match(info.fileName());
				if (!match.hasMatch())
					continue;

				QString key = match.captured(rule.keyCaptureGroup);
				if (key.isEmpty())
					key = datasetFolderName;

				DatasetEntry& entry = result.datasets[key];
				if (entry.datasetFolder.isEmpty())
					entry.datasetFolder = datasetFolder;
				if (entry.filesByRule.contains(ruleIndex)) {
					result.messages << QString("Duplicate match ignored for dataset [%1], rule [%2]: %3").arg(key).arg(ruleDisplayName(rule, ruleIndex)).arg(absPath);
					continue;
				}
				entry.filesByRule[ruleIndex] = absPath;
				++matchedForRule;
				++result.matchedFiles;
			}

			if (matchedForRule > 0)
				result.messages << QString("Dataset folder [%1], rule [%2] matched %3 file(s).").arg(datasetFolderName).arg(ruleDisplayName(rule, ruleIndex)).arg(matchedForRule);
		}
	}
	return result;
}

void DatasetAssemblerWidget::refreshRulesFeedback()
{
	QSignalBlocker blocker(m_rulesTable);
	QSet<QString> labels;
	for (int row = 0; row < m_rulesTable->rowCount(); ++row) {
		QTableWidgetItem* labelItem = m_rulesTable->item(row, 2);
		QTableWidgetItem* regexItem = m_rulesTable->item(row, 4);
		QTableWidgetItem* groupItem = m_rulesTable->item(row, 5);
		QTableWidgetItem* enabledItem = m_rulesTable->item(row, 0);
		const bool enabled = enabledItem != nullptr && enabledItem->checkState() == Qt::Checked;

		QString tooltip;
		QColor normalColor = palette().base().color();
		QColor warningColor(255, 244, 204);
		QColor errorColor(255, 220, 220);

		if (regexItem != nullptr) {
			regexItem->setBackground(normalColor);
			const QString regexText = regexItem->text().trimmed();
			if (enabled && regexText.isEmpty()) {
				regexItem->setBackground(errorColor);
				tooltip += "Empty regex. ";
			}
			else if (enabled) {
				const QRegularExpression regex(regexText);
				if (!regex.isValid()) {
					regexItem->setBackground(errorColor);
					tooltip += QString("Invalid regex: %1. ").arg(regex.errorString());
				}
				else if (groupItem != nullptr && groupItem->text().toInt() > regex.captureCount()) {
					groupItem->setBackground(errorColor);
					tooltip += QString("Key group exceeds regex capture count (%1). ").arg(regex.captureCount());
				}
				else if (groupItem != nullptr) {
					groupItem->setBackground(normalColor);
				}
			}
			else if (groupItem != nullptr) {
				groupItem->setBackground(normalColor);
			}
		}

		if (labelItem != nullptr) {
			labelItem->setBackground(normalColor);
			const QString key = labelItem->text().trimmed().toLower();
			if (enabled && !key.isEmpty()) {
				if (labels.contains(key)) {
					labelItem->setBackground(warningColor);
					tooltip += "Duplicate rule label. ";
				}
				labels.insert(key);
			}
			labelItem->setToolTip(tooltip.trimmed());
		}
		if (regexItem != nullptr)
			regexItem->setToolTip(tooltip.trimmed());
		if (groupItem != nullptr)
			groupItem->setToolTip(tooltip.trimmed());
	}
}

QString DatasetAssemblerWidget::ruleDisplayName(const DatasetRule& _rule, int _index) const
{
	return _rule.label.trimmed().isEmpty() ? QString::number(_index + 1) : _rule.label.trimmed();
}

QStringList DatasetAssemblerWidget::splitPathSegments(const QString& _path) const
{
	return QDir::fromNativeSeparators(_path).split('/', Qt::SkipEmptyParts);
}

bool DatasetAssemblerWidget::rulesUseRelativeFolders(const std::vector<DatasetRule>& _rules) const
{
	for (const DatasetRule& rule : _rules)
		if (rule.enabled && !rule.relativeFolder.trimmed().isEmpty())
			return true;
	return false;
}

bool DatasetAssemblerWidget::folderContainsDatasetContent(const QString& _folderPath, const std::vector<DatasetRule>& _rules) const
{
	for (const DatasetRule& rule : _rules) {
		if (!rule.enabled)
			continue;

		QDir baseDir(_folderPath);
		if (!rule.relativeFolder.trimmed().isEmpty())
			baseDir = QDir(baseDir.filePath(rule.relativeFolder.trimmed()));
		if (!baseDir.exists())
			continue;

		const QRegularExpression regex(rule.regex);
		if (!regex.isValid())
			continue;

		QDirIterator it(baseDir.absolutePath(), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::NoIteratorFlags);
		while (it.hasNext()) {
			const QString absPath = it.next();
			const QFileInfo info(absPath);
			if (regex.match(info.fileName()).hasMatch())
				return true;
		}
	}
	return false;
}

QStringList DatasetAssemblerWidget::discoverDatasetFolders(const QString& _rootFolder, const std::vector<DatasetRule>& _rules) const
{
	QStringList datasetFolders;
	const bool stopAboveRelativeFolders = rulesUseRelativeFolders(_rules);
	QDirIterator it(_rootFolder, QDir::Dirs | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);

	QStringList allDirs;
	allDirs << QFileInfo(_rootFolder).absoluteFilePath();
	while (it.hasNext())
		allDirs << QFileInfo(it.next()).absoluteFilePath();
	std::sort(allDirs.begin(), allDirs.end(), [](const QString& a, const QString& b) { return a.count('/') < b.count('/'); });

	for (const QString& dirPath : allDirs) {
		QDir dir(dirPath);
		const QFileInfoList childDirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
		if (stopAboveRelativeFolders) {
			if (folderContainsDatasetContent(dirPath, _rules))
				datasetFolders << dirPath;
		}
		else if (childDirs.isEmpty() && folderContainsDatasetContent(dirPath, _rules)) {
			datasetFolders << dirPath;
		}
	}

	datasetFolders.removeDuplicates();
	return datasetFolders;
}

QStringList DatasetAssemblerWidget::hierarchySegmentsForDatasetFolder(const QString& _rootFolder, const QString& _datasetFolder) const
{
	QString relativeDir = QDir(QFileInfo(_rootFolder).absoluteFilePath()).relativeFilePath(QFileInfo(_datasetFolder).absoluteFilePath());
	QStringList segments = splitPathSegments(relativeDir);
	if (!segments.isEmpty() && segments.front() == ".")
		segments.removeFirst();
	return segments;
}

QTreeWidgetItem* DatasetAssemblerWidget::ensurePreviewNode(QTreeWidgetItem* _parent, const QString& _label, const QString& _type)
{
	const int childCount = _parent == nullptr ? m_previewTree->topLevelItemCount() : _parent->childCount();
	for (int index = 0; index < childCount; ++index) {
		QTreeWidgetItem* item = _parent == nullptr ? m_previewTree->topLevelItem(index) : _parent->child(index);
		if (item != nullptr && item->text(0) == _label && (_type.isEmpty() || item->text(1) == _type))
			return item;
	}

	QTreeWidgetItem* item = new QTreeWidgetItem(QStringList() << _label << _type);
	if (_parent == nullptr)
		m_previewTree->addTopLevelItem(item);
	else
		_parent->addChild(item);
	return item;
}

void DatasetAssemblerWidget::populatePreviewTree(const QStringList& _roots, const std::vector<DatasetRule>& _rules)
{
	m_previewTree->clear();
	for (const QString& rootFolder : _roots) {
		const QString rootName = QFileInfo(rootFolder).fileName().isEmpty() ? rootFolder : QFileInfo(rootFolder).fileName();
		QTreeWidgetItem* rootItem = ensurePreviewNode(nullptr, QString("%1 [%2]").arg(rootName, rootFolder), "Root");
		const ScanResult scan = scanRootFolder(rootFolder, _rules);
		for (auto it = scan.datasets.begin(); it != scan.datasets.end(); ++it) {
			const QString datasetKey = it.key();
			const DatasetEntry& entry = it.value();
			const QString datasetFolder = entry.datasetFolder.isEmpty() ? rootFolder : entry.datasetFolder;
			const QStringList segments = hierarchySegmentsForDatasetFolder(rootFolder, datasetFolder);

			QTreeWidgetItem* parentItem = rootItem;
			for (const QString& segment : segments)
				parentItem = ensurePreviewNode(parentItem, segment, "Level");

			bool missingRequired = false;
			for (int ruleIndex = 0; ruleIndex < (int)_rules.size(); ++ruleIndex) {
				const DatasetRule& rule = _rules[ruleIndex];
				if (rule.enabled && rule.required && !entry.filesByRule.contains(ruleIndex)) {
					missingRequired = true;
					break;
				}
			}
			QTreeWidgetItem* datasetItem = ensurePreviewNode(parentItem, datasetKey, missingRequired ? "Dataset (incomplete)" : "Dataset");
			for (auto fileIt = entry.filesByRule.begin(); fileIt != entry.filesByRule.end(); ++fileIt) {
				const DatasetRule& rule = _rules[fileIt.key()];
				ensurePreviewNode(datasetItem, QFileInfo(fileIt.value()).fileName(), ruleDisplayName(rule, fileIt.key()));
			}
		}
	}
	m_previewTree->expandAll();
}

void DatasetAssemblerWidget::populateHierarchy(MyMultipleObject* _multipleObject, const std::vector<AssembledDatasetInfo>& _assembledInfos) const
{
	if (_multipleObject == nullptr)
		return;

	_multipleObject->clearHierarchy();
	if (_assembledInfos.empty())
		return;

	std::map<std::string, size_t> nodeByKey;
	for (const AssembledDatasetInfo& info : _assembledInfos) {
		if (info.object == nullptr)
			continue;

		size_t objectIndex = 0;
		bool foundObject = false;
		for (; objectIndex < _multipleObject->nbColors(); ++objectIndex) {
			if (_multipleObject->getObject(objectIndex) == info.object) {
				foundObject = true;
				break;
			}
		}
		if (!foundObject)
			continue;

		const QString rootLabel = QFileInfo(info.rootFolder).fileName().isEmpty() ? info.rootFolder : QFileInfo(info.rootFolder).fileName();
		std::string pathKey = rootLabel.toStdString();
		size_t parentIndex;
		auto rootIt = nodeByKey.find(pathKey);
		if (rootIt == nodeByKey.end()) {
			parentIndex = _multipleObject->addHierarchyNode(rootLabel.toStdString(), "Root", -1);
			nodeByKey[pathKey] = parentIndex;
		}
		else {
			parentIndex = rootIt->second;
		}

		for (int segmentIndex = 0; segmentIndex < info.hierarchySegments.size(); ++segmentIndex) {
			const QString segment = info.hierarchySegments[segmentIndex];
			pathKey += "/" + segment.toStdString();
			auto nodeIt = nodeByKey.find(pathKey);
			if (nodeIt == nodeByKey.end()) {
				parentIndex = _multipleObject->addHierarchyNode(segment.toStdString(), QString("Level %1").arg(segmentIndex + 1).toStdString(), (int)parentIndex);
				nodeByKey[pathKey] = parentIndex;
			}
			else {
				parentIndex = nodeIt->second;
			}
		}

		_multipleObject->attachObjectToHierarchyNode(parentIndex, objectIndex);
	}
}
