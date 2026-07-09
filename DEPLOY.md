# Deploy do FeelFrame no Google Cloud

Este guia explica como rodar o FeelFrame localmente (com e sem Docker) e como
publicá-lo em produção no Google Cloud, com deploy automático via GitHub
Actions. No fim há também uma seção com o caminho alternativo usando
**Firebase Hosting** para o front-end.

Arquitetura recomendada (tudo dentro do Google):

```
GitHub (push na main)
   │  GitHub Actions (.github/workflows/deploy.yml)
   ▼
Artifact Registry  ──▶  Cloud Run (feelframe-backend)   ──▶  Firebase Storage (vídeos/PDFs)
                    ──▶  Cloud Run (feelframe-frontend)       (já configurado no projeto)
                                                          ──▶  MongoDB (Atlas, ver seção 4)
```

- **Back-end** (FastAPI + MediaPipe + DeepFace): container Docker rodando no
  **Cloud Run**.
- **Front-end** (React/Vite): build estático servido por **nginx** dentro de
  outro container no **Cloud Run** (opção recomendada) — ou, alternativamente,
  publicado no **Firebase Hosting** (seção 8).
- **Arquivos** (vídeos processados, PDFs de relatório): continuam no
  **Firebase Storage**, que já é Google Cloud Storage por baixo — nenhuma
  mudança de código necessária aqui.
- **Banco de dados**: MongoDB. O Google não tem um MongoDB gerenciado nativo;
  a opção recomendada é o **MongoDB Atlas** (gratuito, sem trocar nenhuma
  linha de código). Veja a seção 4 para a alternativa 100% GCP.

---

## 1. Mapa de arquivos — o que cada um faz

| Arquivo | Responsabilidade |
|---|---|
| `back-end/Dockerfile` | Constrói a imagem do back-end (Python 3.11 + libs de sistema do OpenCV/MediaPipe + dependências do `requirements.txt`). |
| `back-end/.dockerignore` | Evita copiar `venv/`, `.env` e `credentials/` para dentro da imagem. |
| `front-end/app/Dockerfile` | Build multi-stage: compila o React com Vite (`npm run build`) e serve os arquivos estáticos com nginx. |
| `front-end/app/nginx.conf.template` | Config do nginx (roteamento de SPA). O `${PORT}` é substituído automaticamente pelo entrypoint da imagem oficial do nginx com o valor que o Cloud Run injeta. |
| `front-end/app/.dockerignore` | Evita copiar `node_modules/`, `dist/` e `.env` para dentro da imagem. |
| `docker-compose.yml` (raiz) | Sobe back-end + front-end + MongoDB localmente em containers, para testar antes de ir para o Cloud Run. **Não é usado no deploy em produção.** |
| `.github/workflows/deploy.yml` | Pipeline de CI/CD: builda as duas imagens, publica no Artifact Registry e faz o deploy das duas no Cloud Run a cada push na `main`. |
| `back-end/app/.env.example` | Modelo das variáveis de ambiente do back-end. Copie para `.env` localmente — **nunca** commite o `.env` real. |
| `front-end/app/.env.example` | Modelo das variáveis de ambiente do front-end (idem). |
| `front-end/app/firebase.json` / `.firebaserc` | Config do Firebase Hosting, usados apenas se você optar pelo caminho alternativo da seção 8. |
| `back-end/app/credentials/serviceAccountKey.json` | Credencial do Firebase Admin SDK (já existente, gitignorada). Em produção ela **não** vai dentro da imagem Docker — é injetada pelo Cloud Run via Secret Manager (seção 5.6). |

> **Nota:** os arquivos `back-end/app/.env` e `front-end/app/.env` estavam
> versionados no Git. Isso foi corrigido nesta mudança: eles foram removidos
> do controle de versão (`git rm --cached`, o conteúdo continua no seu disco)
> e adicionados ao `.gitignore`. Os `.env.example` — sem segredos reais —
> ficam versionados como modelo. **Revise o `git status` antes de dar commit**
> para confirmar que nenhum segredo real será commitado.

---

## 2. Rodando localmente sem Docker (desenvolvimento do dia a dia)

Continua igual ao fluxo que você já usa:

```bash
# back-end
cd back-end
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
cp app/.env.example app/.env     # preencha os valores reais
uvicorn app.main:app --reload --port 8080 --app-dir .

# front-end (outro terminal)
cd front-end/app
npm install
cp .env.example .env             # preencha os valores reais
npm run dev
```

O `.vscode/launch.json` do projeto já tem o compound **"Rodar Full Stack
(Front + Back)"** configurado com esse mesmo caminho.

---

## 3. Rodando localmente com Docker

Útil para validar que os `Dockerfile`s funcionam antes de mandar para o
GitHub Actions.

```bash
docker compose up --build
```

- Front-end: http://localhost:3000
- Back-end: http://localhost:8080
- Por padrão sobe com `STORAGE_BACKEND=local` (grava em um volume Docker, não
  precisa de credencial do Firebase). Para testar com Firebase Storage,
  crie um `.env` na **raiz** do repositório com `STORAGE_BACKEND=firebase` e
  `FIREBASE_STORAGE_BUCKET=...`, e garanta que
  `back-end/app/credentials/serviceAccountKey.json` existe (é montado como
  volume somente-leitura pelo `docker-compose.yml`).

---

## 4. Banco de dados: MongoDB

O código já lê a conexão via `MONGO_URI` (`back-end/app/utils/DatabaseConfig.py`),
então qualquer MongoDB acessível pela internet funciona sem mudar código.

### Opção recomendada — MongoDB Atlas (gratuito, zero configuração de infra)

1. Crie uma conta em https://www.mongodb.com/cloud/atlas/register.
2. Crie um cluster **M0 (Free Tier)** — escolha o provedor **Google Cloud** e
   a região mais próxima (ex.: `São Paulo (southamerica-east1)`), para manter
   a latência baixa em relação ao Cloud Run.
3. Em **Database Access**, crie um usuário/senha.
4. Em **Network Access**, libere `0.0.0.0/0` (Cloud Run usa IPs de saída
   dinâmicos no plano padrão) ou configure Cloud NAT + IP estático se quiser
   restringir — para o escopo de um TCC, `0.0.0.0/0` com usuário/senha fortes
   é aceitável.
5. Pegue a connection string em **Connect → Drivers**, algo como:
   `mongodb+srv://usuario:senha@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`
6. Esse valor vai no secret `mongo-uri` do Secret Manager (seção 5.6).

### Alternativa 100% dentro do GCP — MongoDB em uma VM do Compute Engine

Se você quiser que absolutamente tudo rode dentro do Google (sem nenhum
serviço de terceiros), suba um MongoDB numa VM `e2-small` do Compute Engine:

```bash
gcloud compute instances create-with-container mongo-vm \
  --zone=southamerica-east1-a \
  --machine-type=e2-small \
  --container-image=mongo:7 \
  --container-restart-policy=always \
  --tags=mongo-server

gcloud compute firewall-rules create allow-mongo \
  --allow=tcp:27017 \
  --target-tags=mongo-server \
  --source-ranges=0.0.0.0/0   # em produção, restrinja ao IP de saída do Cloud Run
```

Essa opção exige que você mesmo cuide de backup, atualizações de segurança e
autenticação do MongoDB — por isso o Atlas é o caminho recomendado por
padrão.

---

## 5. Configuração do Google Cloud (feita uma única vez)

### 5.1 Pré-requisitos

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) instalado e
  autenticado (`gcloud init`).
- Docker instalado (para testar builds localmente, opcional).
- Um projeto GCP com faturamento ativado. Se o Firebase já está configurado
  (bucket `feelframe-2b682.firebasestorage.app`), você provavelmente já tem
  esse projeto — o ID dele é `feelframe-2b682`. Use-o para não duplicar
  infraestrutura.

```bash
gcloud config set project feelframe-2b682
```

### 5.2 Habilitar as APIs necessárias

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  cloudbuild.googleapis.com
```

### 5.3 Criar o repositório no Artifact Registry

É aqui que as imagens Docker geradas pelo CI/CD ficam armazenadas.

```bash
gcloud artifacts repositories create feelframe \
  --repository-format=docker \
  --location=southamerica-east1 \
  --description="Imagens do FeelFrame (back-end e front-end)"
```

### 5.4 Criar a service account usada pelo GitHub Actions

Essa conta de serviço é quem builda e faz o deploy em seu nome a partir do
GitHub. Ela recebe só as permissões necessárias (princípio do menor
privilégio):

```bash
gcloud iam service-accounts create github-deployer \
  --display-name="GitHub Actions - Deploy FeelFrame"

PROJECT_ID=feelframe-2b682
SA_EMAIL=github-deployer@${PROJECT_ID}.iam.gserviceaccount.com

for ROLE in roles/run.admin roles/artifactregistry.writer \
            roles/iam.serviceAccountUser roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE"
done
```

Gere a chave JSON dessa conta — **esse arquivo é um segredo, trate como
senha**:

```bash
gcloud iam service-accounts keys create github-deployer-key.json \
  --iam-account=${SA_EMAIL}
```

O conteúdo desse arquivo vai para o secret do GitHub `GCP_SA_KEY` (seção 6).
Depois de colar o conteúdo no GitHub, **apague o arquivo local**
(`github-deployer-key.json`) — não o deixe no repositório nem no seu disco
sem necessidade.

> Alternativa mais segura (opcional, para quando quiser reforçar): usar
> [Workload Identity Federation](https://github.com/google-github-actions/auth#setting-up-workload-identity-federation)
> em vez de chave JSON, eliminando a necessidade de guardar uma credencial de
> longa duração no GitHub. Não é coberto aqui para manter o tutorial simples,
> mas é a evolução natural quando o projeto sair do escopo acadêmico.

### 5.5 Onde ficam os tokens/segredos de aplicação (JWT, Mongo, Google OAuth, Firebase)

Eles **não** vão para dentro da imagem Docker nem para variáveis soltas no
workflow — ficam no **Secret Manager**, e o Cloud Run os injeta em runtime.
Crie cada um uma única vez:

```bash
# Gere uma chave forte para o JWT:
python -c "import secrets; print(secrets.token_hex(32))"

echo -n "SUA_CONNECTION_STRING_DO_MONGO_ATLAS" | \
  gcloud secrets create mongo-uri --data-file=-

echo -n "A_CHAVE_FORTE_GERADA_ACIMA" | \
  gcloud secrets create jwt-secret --data-file=-

echo -n "SEU_GOOGLE_CLIENT_ID.apps.googleusercontent.com" | \
  gcloud secrets create google-client-id --data-file=-

# A credencial do Firebase Admin SDK (o mesmo arquivo que hoje fica em
# back-end/app/credentials/serviceAccountKey.json) vira um secret também:
gcloud secrets create firebase-credentials \
  --data-file=back-end/app/credentials/serviceAccountKey.json
```

Dê à service account do Cloud Run (a padrão do Compute Engine, a não ser que
você crie uma dedicada) permissão para ler esses secrets:

```bash
RUNTIME_SA=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')-compute@developer.gserviceaccount.com

for SECRET in mongo-uri jwt-secret google-client-id firebase-credentials; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/secretmanager.secretAccessor"
done
```

O workflow (`.github/workflows/deploy.yml`) já referencia esses quatro nomes
de secret via `--set-secrets` no `gcloud run deploy` — você não precisa
editar o YAML, só criar os secrets acima uma vez. Para atualizar um valor no
futuro (ex.: trocar a senha do Mongo):

```bash
echo -n "NOVA_CONNECTION_STRING" | gcloud secrets versions add mongo-uri --data-file=-
```

e faça um novo deploy (push na `main`, ou `gcloud run deploy` de novo) para
o Cloud Run pegar a versão `latest` do secret.

---

## 6. CI/CD — GitHub Actions

O workflow em `.github/workflows/deploy.yml` roda em todo push na `main` que
toque em `back-end/**` ou `front-end/app/**` (ou manualmente pela aba
**Actions → Deploy para o Google Cloud Run → Run workflow**). Ele:

1. Builda a imagem do back-end e publica no Artifact Registry.
2. Faz o deploy do back-end no Cloud Run, injetando os segredos do Secret
   Manager.
3. Pega a URL pública gerada para o back-end.
4. Builda a imagem do front-end **usando essa URL** como
   `VITE_API_BASE_URL` (variáveis `VITE_*` do Vite são embutidas no bundle
   em tempo de build, por isso o front precisa ser buildado depois do back).
5. Faz o deploy do front-end no Cloud Run.

### Secrets a configurar no GitHub

Vá em **Settings → Secrets and variables → Actions → New repository secret**
no repositório do FeelFrame e crie:

| Secret | Valor |
|---|---|
| `GCP_SA_KEY` | Conteúdo completo do `github-deployer-key.json` gerado na seção 5.4 |
| `GCP_PROJECT_ID` | `feelframe-2b682` (ou o ID do seu projeto GCP) |
| `FIREBASE_STORAGE_BUCKET` | `feelframe-2b682.firebasestorage.app` |
| `GOOGLE_CLIENT_ID` | O Client ID OAuth do Google (mesmo valor usado em `GOOGLE_CLIENT_ID`/`VITE_GOOGLE_CLIENT_ID`) — não é um segredo crítico (é público por natureza no fluxo OAuth), mas fica centralizado aqui |

Os demais valores sensíveis (Mongo URI, JWT secret, credencial do Firebase)
**não** entram como secret do GitHub — eles já estão no Secret Manager
(seção 5.5) e são referenciados pelo nome dentro do próprio `gcloud run
deploy`.

### Primeiro deploy

Depois de configurar os 4 secrets acima:

```bash
git push origin main
```

ou dispare manualmente pela aba **Actions**. Acompanhe o progresso ali — o
job `deploy-backend` roda primeiro, depois `deploy-frontend`. Ao final, o
step **"Mostrar URLs"** imprime as duas URLs públicas do Cloud Run.

---

## 7. Deploy manual (sem CI/CD, útil para testar antes de configurar o GitHub Actions)

```bash
# Back-end
cd back-end
gcloud builds submit --tag southamerica-east1-docker.pkg.dev/feelframe-2b682/feelframe/backend:manual
gcloud run deploy feelframe-backend \
  --image=southamerica-east1-docker.pkg.dev/feelframe-2b682/feelframe/backend:manual \
  --region=southamerica-east1 --allow-unauthenticated \
  --memory=4Gi --cpu=2 --timeout=600 \
  --set-env-vars="DB_NAME=feelFrame,JWT_EXPIRE_HOURS=168,STORAGE_BACKEND=firebase,VIDEO_POOL_WORKERS=2,FIREBASE_CREDENTIALS_PATH=app/credentials/serviceAccountKey.json,FIREBASE_STORAGE_BUCKET=feelframe-2b682.firebasestorage.app" \
  --set-secrets="MONGO_URI=mongo-uri:latest,JWT_SECRET_KEY=jwt-secret:latest,GOOGLE_CLIENT_ID=google-client-id:latest,/app/app/credentials/serviceAccountKey.json=firebase-credentials:latest"

# Front-end (troque BACKEND_URL pela URL impressa pelo comando acima)
cd ../front-end/app
BACKEND_URL=$(gcloud run services describe feelframe-backend --region=southamerica-east1 --format='value(status.url)')
gcloud builds submit \
  --tag southamerica-east1-docker.pkg.dev/feelframe-2b682/feelframe/frontend:manual \
  --substitutions=_VITE_API_BASE_URL=$BACKEND_URL
gcloud run deploy feelframe-frontend \
  --image=southamerica-east1-docker.pkg.dev/feelframe-2b682/feelframe/frontend:manual \
  --region=southamerica-east1 --allow-unauthenticated --memory=256Mi
```

> `gcloud builds submit --tag` não passa build args do Dockerfile por padrão
> — para builds manuais com `VITE_API_BASE_URL`, o caminho mais simples é
> buildar localmente com `docker build --build-arg ...` e usar `docker push`,
> como o workflow do GitHub Actions já faz.

---

## 8. Alternativa: Front-end no Firebase Hosting

Como o projeto já usa Firebase (Storage), publicar o front-end no **Firebase
Hosting** também é uma opção totalmente válida — e mais barata/simples que
Cloud Run para um site estático, com CDN global incluída no plano gratuito.

Os arquivos `front-end/app/firebase.json` e `front-end/app/.firebaserc` já
estão prontos (apontam para o projeto `feelframe-2b682` e fazem o rewrite de
SPA para `index.html`).

```bash
npm install -g firebase-tools
firebase login

cd front-end/app
echo "VITE_API_BASE_URL=https://<URL-DO-BACKEND-NO-CLOUD-RUN>" > .env
echo "VITE_GOOGLE_CLIENT_ID=<seu-client-id>" >> .env

npm run build
firebase deploy --only hosting
```

O back-end continua rodando no Cloud Run normalmente (o Firebase Hosting não
executa Python) — o front-end no Hosting simplesmente chama a URL do Cloud
Run via `VITE_API_BASE_URL`, o que já funciona porque o CORS do FastAPI está
liberado (`back-end/app/main.py`, `allow_origins=["*"]`). Antes de ir para
produção de fato, vale restringir esse `allow_origins` ao domínio final do
Hosting/Cloud Run, em vez de `"*"`.

Para automatizar esse caminho também via CI/CD, existe a action oficial
[`FirebaseExtended/action-hosting-deploy`](https://github.com/FirebaseExtended/action-hosting-deploy) —
não incluída no `deploy.yml` atual para não misturar dois caminhos de deploy
do front-end na mesma pipeline; escolha um dos dois (Cloud Run **ou**
Firebase Hosting) e adapte o workflow conforme a escolha.

---

## 9. Variáveis de ambiente — referência completa

### Back-end (`back-end/app/.env` local / Secret Manager + `--set-env-vars` em produção)

| Variável | Onde configurar em produção | Observação |
|---|---|---|
| `MONGO_URI` | Secret Manager (`mongo-uri`) | Connection string do MongoDB Atlas ou da VM |
| `DB_NAME` | `--set-env-vars` | `feelFrame` |
| `JWT_SECRET_KEY` | Secret Manager (`jwt-secret`) | Gere com `secrets.token_hex(32)` |
| `JWT_EXPIRE_HOURS` | `--set-env-vars` | `168` (7 dias) |
| `GOOGLE_CLIENT_ID` | Secret Manager (`google-client-id`) | Do Google Cloud Console → Credenciais OAuth |
| `STORAGE_BACKEND` | `--set-env-vars` | `firebase` em produção |
| `FIREBASE_CREDENTIALS_PATH` | `--set-env-vars` | `app/credentials/serviceAccountKey.json` (caminho onde o secret é montado como arquivo) |
| `FIREBASE_STORAGE_BUCKET` | `--set-env-vars` | `feelframe-2b682.firebasestorage.app` |
| `VIDEO_POOL_WORKERS` | `--set-env-vars` | `2` — mais que isso tende a só aumentar contenção de CPU |
| `OUTPUT_WIDTH` / `OUTPUT_HEIGHT` | `--set-env-vars` | Opcional, default 480x480 |

### Front-end (`front-end/app/.env` local / `--build-arg` no build da imagem)

| Variável | Onde configurar em produção | Observação |
|---|---|---|
| `VITE_API_BASE_URL` | `--build-arg` no build do Docker (CI pega a URL do Cloud Run automaticamente) | Embutida no bundle — mudar exige rebuild |
| `VITE_GOOGLE_CLIENT_ID` | `--build-arg`, valor vindo do secret `GOOGLE_CLIENT_ID` do GitHub | Idem |

---

## 10. Riscos conhecidos e observações importantes

- **Imagem do back-end é grande e pesada** (TensorFlow, PyTorch, MediaPipe,
  DeepFace, OpenCV). O build no CI pode levar vários minutos e a imagem final
  fica na casa de poucos GB — normal para esse stack, o Cloud Run aceita até
  32 GB de imagem. Por isso o deploy usa `--memory=4Gi --cpu=2`; se a
  instância cair por falta de memória durante o processamento de vídeo,
  aumente para `8Gi`.
- **Cold start**: como os modelos de ML são carregados na inicialização do
  container, a primeira requisição depois de um período ocioso pode demorar
  bastante. O Dockerfile já pré-baixa os pesos do modelo de emoção do
  DeepFace **durante o build** (não em runtime) para reduzir isso, mas o
  carregamento do TensorFlow/MediaPipe em si ainda leva alguns segundos. Se
  isso for um problema para a demonstração do TCC, configure
  `--min-instances=1` no `gcloud run deploy` para manter uma instância
  sempre quente (isso gera custo contínuo, ao contrário do padrão
  "escala a zero" do Cloud Run).
- **Codec de vídeo (`avc1`/H.264)**: `back-end/app/services/videoService.py`
  grava os vídeos processados com `cv.VideoWriter_fourcc(*"avc1")`. As wheels
  pip do `opencv-python`/`opencv-contrib-python` no Linux frequentemente
  **não** incluem suporte a codificação H.264 (por questões de licença do
  x264), diferente do Windows, onde o SO fornece o codec via Media
  Foundation. **Teste a geração de vídeo assim que subir para o Cloud Run** —
  se o arquivo de saída vier corrompido/vazio, as saídas mais simples são:
  trocar o fourcc para `mp4v` (sempre funciona, mas verifique a
  compatibilidade de playback no `<video>` do navegador) ou adicionar o
  pacote `imageio-ffmpeg` (traz um binário ffmpeg com libx264 embutido) e
  reescrever a gravação para usá-lo. Isso não foi alterado automaticamente
  aqui porque é uma decisão de produto/qualidade de vídeo, não de infra.
- **`opencv-python` + `opencv-contrib-python` juntos** no
  `requirements.txt`: o pacote `opencv-contrib-python` já inclui tudo que o
  `opencv-python` tem. Ter os dois listados é redundante e, dependendo da
  ordem de instalação, um pode sobrescrever arquivos do outro. Funciona hoje
  porque já é assim no seu ambiente local, mas vale remover o
  `opencv-python` do `requirements.txt` em algum momento para evitar
  builds não determinísticos.
- **`requirements.txt` (raiz do repositório)**: parece ser uma cópia antiga/
  desatualizada de `back-end/requirements.txt` (faltam `bcrypt`, `PyJWT`,
  `firebase_admin`, entre outros) e não é usada por nenhum `Dockerfile` — os
  dois Dockerfiles usam `back-end/requirements.txt`. Considere removê-la para
  evitar confusão futura.
- **CORS liberado para `"*"`** em `back-end/app/main.py`: adequado para
  desenvolvimento, mas depois que as URLs finais do front-end estiverem
  definidas (Cloud Run e/ou Firebase Hosting), restrinja
  `allow_origins` a essas URLs específicas.
- **Custo**: Cloud Run cobra por tempo de CPU/memória realmente usado
  (escala a zero por padrão) + Artifact Registry cobra por armazenamento de
  imagem. Para um projeto de TCC com uso esporádico, o custo tende a ficar
  dentro da faixa gratuita mensal do GCP, exceto se `--min-instances=1` for
  usado continuamente.

---

## 11. Comandos úteis

```bash
# Ver logs em tempo real do back-end
gcloud run services logs tail feelframe-backend --region=southamerica-east1

# Listar revisões e fazer rollback para uma anterior
gcloud run revisions list --service=feelframe-backend --region=southamerica-east1
gcloud run services update-traffic feelframe-backend --region=southamerica-east1 \
  --to-revisions=REVISION_NAME=100

# Ver a URL pública de cada serviço
gcloud run services describe feelframe-backend --region=southamerica-east1 --format='value(status.url)'
gcloud run services describe feelframe-frontend --region=southamerica-east1 --format='value(status.url)'
```
