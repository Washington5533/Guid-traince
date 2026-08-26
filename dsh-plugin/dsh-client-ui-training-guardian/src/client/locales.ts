/**
 * i18n dictionary keys for Training Guardian plugin.
 *
 * The browser half registers this under the 'training-guardian' namespace
 * via ctx.locale.register(NS, { zh, en }).
 */

export type TgKey =
  // Panel
  | 'panel.title'
  | 'panel.noData'
  | 'panel.connecting'
  | 'panel.disconnected'
  | 'panel.idle'
  | 'panel.idleConnect'
  | 'panel.close'
  | 'panel.drag'
  // Connection diagnosis
  | 'conn.failed'
  | 'conn.retry'
  | 'conn.retrying'
  | 'conn.advice'
  | 'conn.adviceBusy'
  | 'conn.attempt'
  | 'conn.unauthorized'
  | 'conn.notFound'
  | 'conn.serverError'
  | 'conn.unreachable'
  | 'conn.exhausted'
  | 'conn.unknown'
  | 'conn.hintUnauthorized'
  | 'conn.hintNotFound'
  | 'conn.hintServerError'
  | 'conn.hintUnreachable'
  // Tabs
  | 'tab.overview'
  | 'tab.gpu'
  | 'tab.anomalies'
  | 'tab.decisions'
  | 'tab.arch'
  // Architecture tab
  | 'arch.title'
  | 'arch.noData'
  | 'arch.loading'
  | 'arch.analyzeBtn'
  | 'arch.error'
  | 'arch.viewTreemap'
  | 'arch.viewBackbone'
  | 'arch.colorParams'
  | 'arch.colorFlops'
  | 'arch.bottlenecks'
  | 'arch.bottleneckCount'
  | 'arch.noBottlenecks'
  | 'arch.params'
  | 'arch.flops'
  | 'arch.repeat'
  | 'arch.subModules'
  | 'arch.moduleCount'
  | 'arch.layerCount'
  | 'arch.elapsedMs'
  // Architecture: agent-driven analysis
  | 'arch.agentBtn'
  | 'arch.agentRunning'
  | 'arch.agentDispatched'
  | 'arch.agentNoSession'
  | 'arch.agentUnavailable'
  | 'arch.modelEntryMissing'
  | 'arch.detectedEntry'
  | 'arch.source'
  | 'arch.sourceAgent'
  | 'arch.sourceDirect'
  // Overview tab
  | 'overview.epoch'
  | 'overview.loss'
  | 'overview.accuracy'
  | 'overview.lr'
  | 'overview.step'
  | 'overview.status'
  | 'overview.noMetrics'
  | 'overview.chartTitle'
  | 'overview.chartLoss'
  | 'overview.chartAccuracy'
  | 'overview.chartClear'
  | 'overview.chartNoData'
  // GPU tab
  | 'gpu.title'
  | 'gpu.utilization'
  | 'gpu.temperature'
  | 'gpu.memory'
  | 'gpu.power'
  | 'gpu.noData'
  // Anomalies tab
  | 'anomalies.title'
  | 'anomalies.none'
  | 'anomalies.type'
  | 'anomalies.description'
  | 'anomalies.time'
  | 'anomalies.severity'
  // Decisions tab
  | 'decisions.title'
  | 'decisions.none'
  | 'decisions.tool'
  | 'decisions.action'
  | 'decisions.source'
  | 'decisions.time'
  | 'decisions.approve'
  | 'decisions.reject'
  // History tab
  | 'tab.history'
  | 'history.title'
  | 'history.noData'
  | 'history.loading'
  | 'history.loadBtn'
  | 'history.back'
  | 'history.sessions'
  | 'history.events'
  | 'history.duration'
  | 'history.metrics'
  | 'history.anomalies'
  | 'history.decisions'
  | 'history.crashes'
  // Panel layout
  | 'panel.layoutFloat'
  | 'panel.layoutSidebar'
  | 'panel.layoutExpanded'
  | 'panel.dashboard'
  // AI analysis & chat
  | 'history.aiAnalyze'
  | 'history.aiAnalyzing'
  | 'history.aiResult'
  | 'history.aiFailed'
  | 'history.aiChat'
  | 'history.aiChatPlaceholder'
  | 'history.aiChatSend'
  | 'history.crashData'
  | 'history.noCrashes'
  | 'history.compare'
  | 'history.compareSelect'
  | 'history.compareRun'
  | 'history.compareResult'
  | 'history.trend'
  // Offline / local history
  | 'history.offlineBanner'
  | 'history.offlineHint'
  | 'history.localBadge'
  | 'history.localNoData'
  | 'history.dataPoints'
  | 'history.localClear'
  // Settings
  | 'settings.title'
  | 'settings.serverUrl'
  | 'settings.serverUrlHint'
  | 'settings.authToken'
  | 'settings.authTokenHint'
  | 'settings.sessionId'
  | 'settings.sessionIdHint'
  | 'settings.autoConnect'
  | 'settings.saved'
  | 'settings.modelEntry'
  | 'settings.modelEntryHint'
  | 'settings.projectDir'
  | 'settings.projectDirHint'
  | 'settings.dashboardUrl'
  | 'settings.dashboardUrlHint'
  // Severity
  | 'severity.low'
  | 'severity.medium'
  | 'severity.high'
  | 'severity.critical'
  // Misc
  | 'misc.unknown'
  | 'misc.never'
  | 'misc.error'
  | 'misc.warning'
  | 'misc.info'

const zh: Record<TgKey, string> = {
  'panel.title': 'Training Guardian',
  'panel.noData': '暂无数据，请检查连接',
  'panel.connecting': '连接中...',
  'panel.disconnected': '已断开',
  'panel.idle': '等待训练任务...',
  'panel.idleConnect': '手动连接',
  'panel.close': '关闭',
  'panel.drag': '拖拽移动',
  // Connection diagnosis
  'conn.failed': '连接失败',
  'conn.retry': '重试',
  'conn.retrying': '正在重试...',
  'conn.advice': 'AI 建议',
  'conn.adviceBusy': '获取建议中...',
  'conn.attempt': '尝试',
  'conn.unauthorized': '认证失败',
  'conn.notFound': '服务未找到',
  'conn.serverError': '服务器错误',
  'conn.unreachable': '无法连接',
  'conn.exhausted': '重试用尽',
  'conn.unknown': '未知错误',
  'conn.hintUnauthorized': '请检查鉴权 Token 是否与 --remote-auth 一致',
  'conn.hintNotFound': '请确认服务器地址和端口是否正确',
  'conn.hintServerError': '服务器内部错误，请查看服务器日志',
  'conn.hintUnreachable': '请检查网络连接和防火墙设置',
  // Tabs
  'tab.overview': '概览',
  'tab.gpu': '设备',
  'tab.anomalies': '异常',
  'tab.decisions': '决策',
  'tab.arch': '架构',
  'overview.epoch': 'Epoch',
  'overview.loss': 'Loss',
  'overview.accuracy': 'Accuracy',
  'overview.lr': 'Learning Rate',
  'overview.step': 'Step',
  'overview.status': '状态',
  'overview.noMetrics': '等待指标数据...',
  'overview.chartTitle': '训练曲线',
  'overview.chartLoss': '损失',
  'overview.chartAccuracy': '准确率',
  'overview.chartClear': '清除历史',
  'overview.chartNoData': '等待数据点以绘制曲线...',
  'gpu.title': 'GPU 设备状态',
  'gpu.utilization': '利用率',
  'gpu.temperature': '温度',
  'gpu.memory': '显存',
  'gpu.power': '功耗',
  'gpu.noData': '暂无 GPU 数据',
  'anomalies.title': '异常事件',
  'anomalies.none': '暂无异常',
  'anomalies.type': '类型',
  'anomalies.description': '描述',
  'anomalies.time': '时间',
  'anomalies.severity': '严重程度',
  'decisions.title': 'Sub-agent 决策',
  'decisions.none': '暂无待审批决策',
  'decisions.tool': '工具',
  'decisions.action': '动作',
  'decisions.source': '来源',
  'decisions.time': '时间',
  'decisions.approve': '批准',
  'decisions.reject': '驳回',
  // Architecture tab
  'arch.title': '模型架构分析',
  'arch.noData': '点击「分析架构」查看模型结构可视化',
  'arch.loading': '分析中...',
  'arch.analyzeBtn': '分析架构',
  'arch.error': '分析失败',
  'arch.viewTreemap': '占比图',
  'arch.viewBackbone': '流水线',
  'arch.colorParams': '按参数',
  'arch.colorFlops': '按 FLOPs',
  'arch.bottlenecks': '瓶颈层',
  'arch.bottleneckCount': '个瓶颈',
  'arch.noBottlenecks': '未检测到瓶颈层',
  'arch.params': '参数量',
  'arch.flops': 'FLOPs',
  'arch.repeat': '重复',
  'arch.subModules': '子模块',
  'arch.moduleCount': '模块数',
  'arch.layerCount': '层数',
  'arch.elapsedMs': '耗时',
  // Architecture: agent-driven analysis
  'arch.agentBtn': 'AI 分析架构',
  'arch.agentRunning': 'AI 正在分析...',
  'arch.agentDispatched': '已发送给 AI',
  'arch.agentNoSession': '无可用会话',
  'arch.agentUnavailable': 'AI 不可用',
  'arch.modelEntryMissing': '请在设置中配置模型入口 (model_entry)',
  'arch.detectedEntry': '已检测到入口',
  'arch.source': '来源',
  'arch.sourceAgent': 'AI 生成',
  'arch.sourceDirect': '直接分析',
  // History tab
  'tab.history': '历史',
  'history.title': '历史会话',
  'history.noData': '暂无历史训练记录',
  'history.loading': '加载中...',
  'history.loadBtn': '查看',
  'history.back': '返回实时',
  'history.sessions': '个会话',
  'history.events': '事件',
  'history.duration': '时长',
  'history.metrics': '指标',
  'history.anomalies': '异常',
  'history.decisions': '决策',
  'history.crashes': '崩溃',
  // AI analysis & chat
  'history.aiAnalyze': 'AI 分析',
  'history.aiAnalyzing': 'AI 分析中...',
  'history.aiResult': 'AI 解读',
  'history.aiFailed': 'AI 分析失败',
  'history.aiChat': 'AI 追问',
  'history.aiChatPlaceholder': '输入关于这次训练的问题...',
  'history.aiChatSend': '发送',
  'history.crashData': '崩溃记录',
  'history.noCrashes': '无崩溃记录',
  'history.compare': '对比',
  'history.compareSelect': '选择要对比的会话',
  'history.compareRun': '开始对比',
  'history.compareResult': '对比结果',
  'history.trend': '趋势',
  // Offline / local history
  'history.offlineBanner': '服务器未连接，以下为本地缓存的记录',
  'history.offlineHint': '启动 RemoteServer 可查看完整历史记录',
  'history.localBadge': '本地',
  'history.localNoData': '暂无本地缓存的训练记录',
  'history.dataPoints': '数据点',
  'history.localClear': '清除缓存',
  // Panel layout
  'panel.layoutFloat': '浮动',
  'panel.layoutSidebar': '侧栏',
  'panel.layoutExpanded': '全屏',
  'panel.dashboard': '打开 Dashboard',
  // Settings
  'settings.title': 'Training Guardian',
  'settings.serverUrl': 'Guardian 服务器地址',
  'settings.serverUrlHint': '算力服务器的 RemoteServer 地址，如 http://192.168.1.100:8765',
  'settings.authToken': '鉴权 Token',
  'settings.authTokenHint': '可选，与 --remote-auth 一致',
  'settings.sessionId': '训练会话 ID',
  'settings.sessionIdHint': '留空则自动连接第一个活跃会话',
  'settings.autoConnect': '自动连接',
  'settings.saved': '已保存',
  'settings.modelEntry': '模型入口',
  'settings.modelEntryHint': '格式 module:function，如 scripts.train:build_model',
  'settings.projectDir': '项目目录',
  'settings.projectDirHint': '训练项目根目录的绝对路径',
  'settings.dashboardUrl': 'Dashboard 地址',
  'settings.dashboardUrlHint': 'Guardian Dashboard 页面地址，如 http://192.168.1.100:8765',
  // Severity
  'severity.low': '低',
  'severity.medium': '中',
  'severity.high': '高',
  'severity.critical': '紧急',
  'misc.unknown': '未知',
  'misc.never': '从未',
  'misc.error': '错误',
  'misc.warning': '警告',
  'misc.info': '信息',
}

const en: Record<TgKey, string> = {
  'panel.title': 'Training Guardian',
  'panel.noData': 'No data — check connection',
  'panel.connecting': 'Connecting...',
  'panel.disconnected': 'Disconnected',
  'panel.idle': 'Waiting for training task...',
  'panel.idleConnect': 'Connect',
  'panel.close': 'Close',
  'panel.drag': 'Drag to move',
  // Connection diagnosis
  'conn.failed': 'Connection failed',
  'conn.retry': 'Retry',
  'conn.retrying': 'Retrying...',
  'conn.advice': 'AI Advice',
  'conn.adviceBusy': 'Getting advice...',
  'conn.attempt': 'Attempt',
  'conn.unauthorized': 'Authentication failed',
  'conn.notFound': 'Service not found',
  'conn.serverError': 'Server error',
  'conn.unreachable': 'Cannot reach server',
  'conn.exhausted': 'Retries exhausted',
  'conn.unknown': 'Unknown error',
  'conn.hintUnauthorized': 'Check that the auth token matches --remote-auth',
  'conn.hintNotFound': 'Verify the server address and port',
  'conn.hintServerError': 'Check the server logs for details',
  'conn.hintUnreachable': 'Check your network and firewall settings',
  // Tabs
  'tab.overview': 'Overview',
  'tab.gpu': 'Devices',
  'tab.anomalies': 'Anomalies',
  'tab.decisions': 'Decisions',
  'tab.arch': 'Architecture',
  'overview.epoch': 'Epoch',
  'overview.loss': 'Loss',
  'overview.accuracy': 'Accuracy',
  'overview.lr': 'LR',
  'overview.step': 'Step',
  'overview.status': 'Status',
  'overview.noMetrics': 'Waiting for metrics...',
  'overview.chartTitle': 'Training Curves',
  'overview.chartLoss': 'Loss',
  'overview.chartAccuracy': 'Accuracy',
  'overview.chartClear': 'Clear History',
  'overview.chartNoData': 'Waiting for data points to plot...',
  'gpu.title': 'GPU Status',
  'gpu.utilization': 'Utilization',
  'gpu.temperature': 'Temperature',
  'gpu.memory': 'VRAM',
  'gpu.power': 'Power',
  'gpu.noData': 'No GPU data',
  'anomalies.title': 'Anomalies',
  'anomalies.none': 'No anomalies',
  'anomalies.type': 'Type',
  'anomalies.description': 'Description',
  'anomalies.time': 'Time',
  'anomalies.severity': 'Severity',
  'decisions.title': 'Sub-agent Decisions',
  'decisions.none': 'No pending decisions',
  'decisions.tool': 'Tool',
  'decisions.action': 'Action',
  'decisions.source': 'Source',
  'decisions.time': 'Time',
  'decisions.approve': 'Approve',
  'decisions.reject': 'Reject',
  // Architecture tab
  'arch.title': 'Architecture Analysis',
  'arch.noData': 'Click "Analyze" to visualize model architecture',
  'arch.loading': 'Analyzing...',
  'arch.analyzeBtn': 'Analyze',
  'arch.error': 'Analysis failed',
  'arch.viewTreemap': 'Treemap',
  'arch.viewBackbone': 'Flow',
  'arch.colorParams': 'By Params',
  'arch.colorFlops': 'By FLOPs',
  'arch.bottlenecks': 'Bottlenecks',
  'arch.bottleneckCount': 'bottlenecks',
  'arch.noBottlenecks': 'No bottlenecks detected',
  'arch.params': 'Parameters',
  'arch.flops': 'FLOPs',
  'arch.repeat': 'Repeat',
  'arch.subModules': 'Sub-modules',
  'arch.moduleCount': 'Modules',
  'arch.layerCount': 'Layers',
  'arch.elapsedMs': 'Elapsed',
  // Architecture: agent-driven analysis
  'arch.agentBtn': 'AI Analyze',
  'arch.agentRunning': 'AI analyzing...',
  'arch.agentDispatched': 'Sent to AI',
  'arch.agentNoSession': 'No session available',
  'arch.agentUnavailable': 'AI unavailable',
  'arch.modelEntryMissing': 'Configure model entry in settings',
  'arch.detectedEntry': 'Entry detected',
  'arch.source': 'Source',
  'arch.sourceAgent': 'AI-generated',
  'arch.sourceDirect': 'Direct analysis',
  // History tab
  'tab.history': 'History',
  'history.title': 'Past Sessions',
  'history.noData': 'No historical training sessions found',
  'history.loading': 'Loading...',
  'history.loadBtn': 'View',
  'history.back': 'Back to Live',
  'history.sessions': 'sessions',
  'history.events': 'events',
  'history.duration': 'duration',
  'history.metrics': 'metrics',
  'history.anomalies': 'anomalies',
  'history.decisions': 'decisions',
  'history.crashes': 'crashes',
  // AI analysis & chat
  'history.aiAnalyze': 'AI Analysis',
  'history.aiAnalyzing': 'Analyzing...',
  'history.aiResult': 'AI Analysis',
  'history.aiFailed': 'AI analysis failed',
  'history.aiChat': 'AI Chat',
  'history.aiChatPlaceholder': 'Ask a question about this training...',
  'history.aiChatSend': 'Send',
  'history.crashData': 'Crash Records',
  'history.noCrashes': 'No crash records',
  'history.compare': 'Compare',
  'history.compareSelect': 'Select sessions to compare',
  'history.compareRun': 'Compare',
  'history.compareResult': 'Comparison',
  'history.trend': 'Trend',
  // Offline / local history
  'history.offlineBanner': 'Server offline — showing locally cached records',
  'history.offlineHint': 'Start RemoteServer to view full history',
  'history.localBadge': 'Local',
  'history.localNoData': 'No locally cached training records',
  'history.dataPoints': 'data points',
  'history.localClear': 'Clear cache',
  // Panel layout
  'panel.layoutFloat': 'Float',
  'panel.layoutSidebar': 'Sidebar',
  'panel.layoutExpanded': 'Expanded',
  'panel.dashboard': 'Open Dashboard',
  // Settings
  'settings.title': 'Training Guardian',
  'settings.serverUrl': 'Guardian Server URL',
  'settings.serverUrlHint': 'e.g. http://192.168.1.100:8765',
  'settings.authToken': 'Auth Token',
  'settings.authTokenHint': 'Optional, must match --remote-auth',
  'settings.sessionId': 'Session ID',
  'settings.sessionIdHint': 'Leave empty to auto-connect to first active session',
  'settings.autoConnect': 'Auto Connect',
  'settings.saved': 'Saved',
  'settings.modelEntry': 'Model Entry',
  'settings.modelEntryHint': 'Format module:function, e.g. scripts.train:build_model',
  'settings.projectDir': 'Project Directory',
  'settings.projectDirHint': 'Absolute path to the training project root',
  'settings.dashboardUrl': 'Dashboard URL',
  'settings.dashboardUrlHint': 'Guardian Dashboard page URL, e.g. http://192.168.1.100:8765',
  // Severity
  'severity.low': 'Low',
  'severity.medium': 'Medium',
  'severity.high': 'High',
  'severity.critical': 'Critical',
  'misc.unknown': 'Unknown',
  'misc.never': 'Never',
  'misc.error': 'Error',
  'misc.warning': 'Warning',
  'misc.info': 'Info',
}

export { zh, en }
