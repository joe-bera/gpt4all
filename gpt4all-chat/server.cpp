#include "server.h"
#include "llm.h"
#include "download.h"
#include "chatlistmodel.h"

#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <iostream>

#define DEBUG

static inline QString modelToName(const ModelInfo &info)
{
    QString modelName = info.filename;
    Q_ASSERT(modelName.startsWith("ggml-"));
    modelName = modelName.remove(0, 5);
    Q_ASSERT(modelName.endsWith(".bin"));
    modelName.chop(4);
    return modelName;
}

static inline QJsonObject modelToJson(const ModelInfo &info)
{
    QString modelName = modelToName(info);

    QJsonObject model;
    model.insert("id", modelName);
    model.insert("object", "model");
    model.insert("created", "who can keep track?");
    model.insert("owned_by", "humanity");
    model.insert("root", modelName);
    model.insert("parent", QJsonValue::Null);

    QJsonArray permissions;
    QJsonObject permissionObj;
    permissionObj.insert("id", "foobarbaz");
    permissionObj.insert("object", "model_permission");
    permissionObj.insert("created", "does it really matter?");
    permissionObj.insert("allow_create_engine", false);
    permissionObj.insert("allow_sampling", false);
    permissionObj.insert("allow_logprobs", false);
    permissionObj.insert("allow_search_indices", false);
    permissionObj.insert("allow_view", true);
    permissionObj.insert("allow_fine_tuning", false);
    permissionObj.insert("organization", "*");
    permissionObj.insert("group", QJsonValue::Null);
    permissionObj.insert("is_blocking", false);
    permissions.append(permissionObj);
    model.insert("permissions", permissions);
    return model;
}

Server::Server(Chat *chat)
    : ChatLLM(chat)
    , m_chat(chat)
    , m_server(nullptr)
{
    connect(this, &Server::threadStarted, this, &Server::start);
}

Server::~Server()
{
}

void Server::start()
{
    m_server = new QHttpServer(this);
    //connect(m_server, &QTcpServer::newConnection, this, &Server::handleNewConnection);
    if (!m_server->listen(QHostAddress::LocalHost, 4891)) {
        qWarning() << "ERROR: Unable to start the server";
        return;
    }

    m_server->route("/v1/models", QHttpServerRequest::Method::Get,
        [](const QHttpServerRequest &request) {
            const QList<ModelInfo> modelList = Download::globalInstance()->modelList();
            QJsonObject root;
            root.insert("object", "list");
            QJsonArray data;
            for (const ModelInfo &info : modelList) {
                if (!info.installed)
                    continue;
                data.append(modelToJson(info));
            }
            root.insert("data", data);
            return QHttpServerResponse(root);
        }
    );

    m_server->route("/v1/models/<arg>", QHttpServerRequest::Method::Get,
        [](const QString &model, const QHttpServerRequest &request) {
            const QList<ModelInfo> modelList = Download::globalInstance()->modelList();
            QJsonObject object;
            for (const ModelInfo &info : modelList) {
                if (!info.installed)
                    continue;

                QString modelName = modelToName(info);
                if (model == modelName) {
                    object = modelToJson(info);
                    break;
                }
            }
            return QHttpServerResponse(object);
        }
    );

    m_server->route("/v1/completions", QHttpServerRequest::Method::Post,
        [=](const QHttpServerRequest &request) {
            return handleCompletionRequest(request);
        }
    );

    connect(this, &Server::requestServerNewPromptResponsePair, m_chat,
        &Chat::serverNewPromptResponsePair, Qt::BlockingQueuedConnection);
}

QHttpServerResponse Server::handleCompletionRequest(const QHttpServerRequest &request)
{
    // We've been asked to do a completion...
    QJsonParseError err;
    const QJsonDocument document = QJsonDocument::fromJson(request.body(), &err);
    if (err.error || !document.isObject()) {
        std::cerr << "ERROR: invalid json in completions body" << std::endl;
        return QHttpServerResponse(QHttpServerResponder::StatusCode::NoContent);
    }
#if defined(DEBUG)
    printf("/v1/completions %s\n", qPrintable(document.toJson(QJsonDocument::Indented)));
    fflush(stdout);
#endif
    const QJsonObject body = document.object();
    if (!body.contains("model")) { // required
        std::cerr << "ERROR: completions contains no model" << std::endl;
        return QHttpServerResponse(QHttpServerResponder::StatusCode::NoContent);
    }

    const QString model = body["model"].toString();
    bool foundModel = false;
    const QList<ModelInfo> modelList = Download::globalInstance()->modelList();
    for (const ModelInfo &info : modelList) {
        if (!info.installed)
            continue;
        if (model == modelToName(info)) {
            foundModel = true;
            break;
        }
    }

    if (!foundModel) {
        std::cerr << "ERROR: couldn't find model for completion" << model.toStdString() << std::endl;
        return QHttpServerResponse(QHttpServerResponder::StatusCode::BadRequest);
    }

    if (!loadModel(model)) {
        std::cerr << "ERROR: couldn't load model" << model.toStdString() << std::endl;
        return QHttpServerResponse(QHttpServerResponder::StatusCode::InternalServerError);
    }

    QList<QString> prompts;
    if (body.contains("prompt")) {
        QJsonValue promptValue = body["prompt"];
        if (promptValue.isString())
            prompts.append(promptValue.toString());
        else {
            QJsonArray array = promptValue.toArray();
            for (QJsonValue v : array)
                prompts.append(v.toString());
        }
    } else
        prompts.append("<|endoftext|>"); // FIXME: Should this be a null string?

    QString suffix;
    if (body.contains("suffix"))
        suffix = body["suffix"].toString();

    int max_tokens = 16;
    if (body.contains("max_tokens"))
        max_tokens = body["max_tokens"].toInt();

    float temperature = 1.f;
    if (body.contains("temperature"))
        temperature = body["temperature"].toDouble();

    float top_p = 1.f;
    if (body.contains("top_p"))
        top_p = body["top_p"].toDouble();

    int n = 1;
    if (body.contains("n"))
        n = body["n"].toInt();

    bool stream = false;
    if (body.contains("stream"))
        stream = body["stream"].toBool();

    int logprobs = -1; // supposed to be null by default??
    if (body.contains("logprobs"))
        logprobs = body["logprobs"].toInt();

    bool echo = false;
    if (body.contains("echo"))
        echo = body["echo"].toBool();

    QList<QString> stop;
    if (body.contains("stop")) {
        QJsonValue stopValue = body["stop"];
        if (stopValue.isString())
            stop.append(stopValue.toString());
        else {
            QJsonArray array = stopValue.toArray();
            for (QJsonValue v : array)
                stop.append(v.toString());
        }
    }

    float presence_penalty = 0.f;
    if (body.contains("presence_penalty"))
        top_p = body["presence_penalty"].toDouble();

    float frequency_penalty = 0.f;
    if (body.contains("frequency_penalty"))
        top_p = body["frequency_penalty"].toDouble();

    int best_of = 1;
    if (body.contains("best_of"))
        logprobs = body["best_of"].toInt();

    //FIXME: how to handle logit_bias?

    QString user;
    if (body.contains("user"))
        suffix = body["user"].toString();

    // don't remember any context
    resetContextProtected();

    // adds prompt/response items to GUI
    emit requestServerNewPromptResponsePair(prompts.first()); // blocks

    // FIXME: need to get from settings those that aren't specified
    // FIXME: need to translate from their settings to ours
    if (!prompt(prompts.first(),
        "%1" /*prompt template*/,
        max_tokens /*n_predict*/,
        m_ctx.top_k /*top_k*/,
        top_p,
        temperature,
        m_ctx.n_batch /*n_batch*/,
        m_ctx.repeat_penalty /*repeat_penalty*/,
        m_ctx.repeat_last_n /*repeat_penalty_tokens*/,
        LLM::globalInstance()->threadCount())) {

        std::cerr << "ERROR: couldn't prompt model" << model.toStdString() << std::endl;
        return QHttpServerResponse(QHttpServerResponder::StatusCode::InternalServerError);
    }

    QJsonObject responseObject;
    responseObject.insert("id", "foobarbaz");
    responseObject.insert("object", "text_completion");
    responseObject.insert("created", QDateTime::currentSecsSinceEpoch());
    responseObject.insert("model", model);

    QJsonArray choices;
    QJsonObject choice;
    choice.insert("text", response());
    choice.insert("index", 0);
    choice.insert("logprobs", QJsonValue::Null);
    choice.insert("finish_reason", "stop"); // FIXME
    choices.append(choice);
    responseObject.insert("choices", choices);

    QJsonObject usage;

    usage.insert("prompt_tokens", int(m_promptTokens));
    usage.insert("completion_tokens", int(m_promptResponseTokens - m_promptTokens));
    usage.insert("total_tokens", int(m_promptResponseTokens));
    responseObject.insert("usage", usage);

#if defined(DEBUG)
    QJsonDocument newDoc(responseObject);
    printf("/v1/completions %s\n", qPrintable(newDoc.toJson(QJsonDocument::Indented)));
    fflush(stdout);
#endif

    return QHttpServerResponse(responseObject);
}
