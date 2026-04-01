/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DatasetAssemblerWidget.hpp
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

#ifndef DatasetAssemblerWidget_h__
#define DatasetAssemblerWidget_h__

#include <QtWidgets/QWidget>
#include <QtCore/QMap>
#include <QtCore/QStringList>
#include <General/json.hpp>

class QListWidget;
class QPushButton;
class QTableWidget;
class QTextEdit;
class QLineEdit;
class QCheckBox;
class QTreeWidget;
class QTreeWidgetItem;

namespace poca::core {
	class MyObjectInterface;
}

class DatasetAssemblerWidget : public QWidget
{
	Q_OBJECT

public:
	DatasetAssemblerWidget(QWidget* = nullptr);
	~DatasetAssemblerWidget() = default;

	void loadParameters(const nlohmann::json&);
	void saveParameters(nlohmann::json&) const;

signals:
	void transferNewObjectCreated(poca::core::MyObjectInterface*);

private slots:
	void onAddFolder();
	void onRemoveFolder();
	void onAddRule();
	void onRemoveRule();
	void onImportJson();
	void onExportJson();
	void onPreview();
	void onAssemble();
	void onRulesChanged();

private:
	struct DatasetRule {
		bool enabled{ true };
		bool required{ true };
		QString label;
		QString relativeFolder;
		QString regex;
		int keyCaptureGroup{ 1 };
	};

	struct DatasetEntry {
		QString datasetFolder;
		QMap<int, QString> filesByRule;
	};

	struct ScanResult {
		QMap<QString, DatasetEntry> datasets;
		QStringList messages;
		int matchedFiles{ 0 };
	};

	struct AssembledDatasetInfo {
		QString rootFolder;
		QString datasetKey;
		QString objectName;
		QStringList hierarchySegments;
		poca::core::MyObjectInterface* object{ nullptr };
	};

	void appendLog(const QString&);
	std::vector<DatasetRule> rulesFromTable() const;
	void setRulesToTable(const std::vector<DatasetRule>&);
	DatasetRule defaultRule() const;
	bool validateConfiguration(const std::vector<DatasetRule>&, const QStringList&, QStringList&, QStringList&) const;
	ScanResult scanRootFolder(const QString&, const std::vector<DatasetRule>&) const;
	void refreshRulesFeedback();
	QString ruleDisplayName(const DatasetRule&, int) const;
	QStringList splitPathSegments(const QString&) const;
	QStringList hierarchySegmentsForDatasetFolder(const QString&, const QString&) const;
	QStringList discoverDatasetFolders(const QString&, const std::vector<DatasetRule>&) const;
	bool rulesUseRelativeFolders(const std::vector<DatasetRule>&) const;
	bool folderContainsDatasetContent(const QString&, const std::vector<DatasetRule>&) const;
	void populatePreviewTree(const QStringList&, const std::vector<DatasetRule>&);
	QTreeWidgetItem* ensurePreviewNode(QTreeWidgetItem*, const QString&, const QString& = QString());
	void populateHierarchy(class MyMultipleObject*, const std::vector<AssembledDatasetInfo>&) const;

	QListWidget* m_rootsList{ nullptr };
	QPushButton* m_addFolderButton{ nullptr };
	QPushButton* m_removeFolderButton{ nullptr };
	QPushButton* m_addRuleButton{ nullptr };
	QPushButton* m_removeRuleButton{ nullptr };
	QPushButton* m_importJsonButton{ nullptr };
	QPushButton* m_exportJsonButton{ nullptr };
	QPushButton* m_previewButton{ nullptr };
	QPushButton* m_assembleButton{ nullptr };
	QTableWidget* m_rulesTable{ nullptr };
	QTreeWidget* m_previewTree{ nullptr };
	QTextEdit* m_logEdit{ nullptr };
	QLineEdit* m_nameSeparatorEdit{ nullptr };
	QCheckBox* m_prefixRootNameCBox{ nullptr };
	QString m_lastRootPath;
};

#endif
