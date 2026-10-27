//! Matriz Sexagesimal de Fase (Base 60) — Roda Cognitiva Sumeriana de Blake-Aeon.
//!
//! 6 Reinos Gravitacionais x 10 Casas = 60 Graus de Liberdade no anel Z_60.
//! Operacao em memoria estatica contigua com zero alocacao dinamica.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Realm {
    VacuoPrimordial = 1,  // 0° - 59°: Silencio, Inercia e Espaco Potencial
    TensaoRuptura = 2,    // 60° - 119°: Pulso, Pressao e Ignicao
    ForjaCoesao = 3,      // 120° - 179°: Estruturacao, Tensores e Leis
    RessonanciaFluxo = 4, // 180° - 239°: Comunhao, Harmonia e Semantica
    FogoVontade = 5,      // 240° - 299°: Dialetica, Decisao e Mutacao
    SeloTempo = 6,        // 300° - 359°: Sintese, Ouroboros e Apoptose
}

impl Realm {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::VacuoPrimordial => "Reino I: O Vacuo Primordial",
            Self::TensaoRuptura => "Reino II: A Tensao e Ruptura",
            Self::ForjaCoesao => "Reino III: A Forja e Coesao",
            Self::RessonanciaFluxo => "Reino IV: A Ressonancia e Fluxo",
            Self::FogoVontade => "Reino V: O Fogo e a Vontade",
            Self::SeloTempo => "Reino VI: O Selo e o Tempo",
        }
    }

    pub const fn color(&self) -> &'static str {
        match self {
            Self::VacuoPrimordial => "#38bdf8",
            Self::TensaoRuptura => "#fb923c",
            Self::ForjaCoesao => "#818cf8",
            Self::RessonanciaFluxo => "#34d399",
            Self::FogoVontade => "#f87171",
            Self::SeloTempo => "#c084fc",
        }
    }

    pub const fn from_id(id: u8) -> Self {
        match id {
            1 => Self::VacuoPrimordial,
            2 => Self::TensaoRuptura,
            3 => Self::ForjaCoesao,
            4 => Self::RessonanciaFluxo,
            5 => Self::FogoVontade,
            _ => Self::SeloTempo,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct House {
    pub id: u8,
    pub realm: Realm,
    pub deg: u16,
    pub theta: f64,
    pub symbol: &'static str,
    pub name: &'static str,
    pub sanskrit_sumerian: &'static str,
    pub action: &'static str,
}

macro_rules! house {
    ($id:expr, $realm:ident, $deg:expr, $sym:expr, $name:expr, $glyph:expr, $act:expr) => {
        House {
            id: $id,
            realm: Realm::$realm,
            deg: $deg,
            theta: ($deg as f64) * std::f64::consts::PI / 180.0,
            symbol: $sym,
            name: $name,
            sanskrit_sumerian: $glyph,
            action: $act,
        }
    };
}

pub static HOUSES: [House; 60] = [
    // --- REINO I: O VACUO PRIMORDIAL (0° - 54°) ---
    house!(
        1,
        VacuoPrimordial,
        0,
        "🌌",
        "O Ponto Imovel",
        "Nir-Bindu",
        "Zera acumuladores de gradiente em SHM; ancora repouso"
    ),
    house!(
        2,
        VacuoPrimordial,
        6,
        "✨",
        "A Primeira Tremula",
        "Spanda",
        "Leitura de temperatura e ruido acustico residual"
    ),
    house!(
        3,
        VacuoPrimordial,
        12,
        "🕳️",
        "A Fenda Obvia",
        "Kha",
        "Mapeamento sem alocacao de descritores em /dev/shm"
    ),
    house!(
        4,
        VacuoPrimordial,
        18,
        "🌱",
        "A Matriz Latente",
        "Yoni-Garbha",
        "Bloqueio mlock em buffers de fase para prevenir swap"
    ),
    house!(
        5,
        VacuoPrimordial,
        24,
        "🌬️",
        "O Alento Retido",
        "Prana-Stasis",
        "Sincronizacao de barreira atomica entre threads"
    ),
    house!(
        6,
        VacuoPrimordial,
        30,
        "🪞",
        "O Espelho Negro",
        "Adarsa",
        "Snapshot /proc/pressure e memoria fisica do Debian"
    ),
    house!(
        7,
        VacuoPrimordial,
        36,
        "🪐",
        "A Semente Gravitacional",
        "Bija",
        "Injecao de pesos centrais no tensor de Blake"
    ),
    house!(
        8,
        VacuoPrimordial,
        42,
        "🧭",
        "A Orientacao Implicita",
        "Disa-Gati",
        "Alinhamento das mascaras AVX-512 com dimensoes"
    ),
    house!(
        9,
        VacuoPrimordial,
        48,
        "⚡",
        "O Potencial Eletrostatico",
        "Vidyut-Kosa",
        "Verificacao de voltagem e frequencia do clock"
    ),
    house!(
        10,
        VacuoPrimordial,
        54,
        "🚪",
        "O Limiar do Primeiro Reino",
        "Dvara-Sandhi",
        "Verificacao de integridade antes da primeira ignicao"
    ),
    // --- REINO II: A TENSAO E RUPTURA (60° - 114°) ---
    house!(
        11,
        TensaoRuptura,
        60,
        "🔥",
        "A Centelha Ruptora",
        "Agni-Sphulinga",
        "Disparo do primeiro pulso no canal sinaptico IPC"
    ),
    house!(
        12,
        TensaoRuptura,
        66,
        "🏹",
        "O Arco Retesado",
        "Dhanu-Tana",
        "Enfileiramento dos tensores de atencao no pipeline"
    ),
    house!(
        13,
        TensaoRuptura,
        72,
        "⚡",
        "O Raio Perfurante",
        "Vajra-Pata",
        "Execucao da primeira camada de projecao linear L0"
    ),
    house!(
        14,
        TensaoRuptura,
        78,
        "🌋",
        "A Fissura Vulcanica",
        "Bhumi-Bheda",
        "Ruptura de limiar de ativacao em pesos densos"
    ),
    house!(
        15,
        TensaoRuptura,
        84,
        "🌀",
        "O Vortice Acelerador",
        "Cakra-Varta",
        "Aceleracao vetorial AVX-VNNI em blocos quantizados"
    ),
    house!(
        16,
        TensaoRuptura,
        90,
        "⚖️",
        "A Ortogonalidade Critica",
        "Sama-Kona",
        "Normalizacao ortogonal de Schmidt nos vetores 640D"
    ),
    house!(
        17,
        TensaoRuptura,
        96,
        "🛡️",
        "O Escudo Deformado",
        "Carma-Vyakoce",
        "Buffer elastico absorvendo picos de jitter no kernel"
    ),
    house!(
        18,
        TensaoRuptura,
        102,
        "🩸",
        "O Sangue do Processador",
        "Rasa-Dhatu",
        "Transmissao direta via barramento de memoria L3"
    ),
    house!(
        19,
        TensaoRuptura,
        108,
        "🔨",
        "O Golpe de Forja",
        "Kuta-Prahara",
        "Quantizacao INT8 em tensores sem perda de fidelidade"
    ),
    house!(
        20,
        TensaoRuptura,
        114,
        "💎",
        "A Fratura Transparente",
        "Sphatika-Bheda",
        "Filtro de atenuacao de ruido termico na APU"
    ),
    // --- REINO III: A FORJA E COESAO (120° - 174°) ---
    house!(
        21,
        ForjaCoesao,
        120,
        "📐",
        "O Compasso Imutavel",
        "Mana-Yantra",
        "Fixacao da geometria 6D no espaco projetivo"
    ),
    house!(
        22,
        ForjaCoesao,
        126,
        "🧱",
        "O Tijolo Alquimico",
        "Istaka-Bandha",
        "Consolidacao do estado eventual no Postgres swa_matrix"
    ),
    house!(
        23,
        ForjaCoesao,
        132,
        "🏛️",
        "A Coluna Central",
        "Stambha-Mula",
        "Ancoragem do atrator da Persona Mestre no disco"
    ),
    house!(
        24,
        ForjaCoesao,
        138,
        "🕸️",
        "A Trama Hexagonal",
        "Madhu-Jala",
        "Conexao em anel entre os 6 eixos de Blake"
    ),
    house!(
        25,
        ForjaCoesao,
        144,
        "⛓️",
        "O No Inextricavel",
        "Granthi-Drdha",
        "Preservacao de chaves KV atemporais no cache"
    ),
    house!(
        26,
        ForjaCoesao,
        150,
        "🌊",
        "O Leito da Torrente",
        "Nadi-Marga",
        "Abertura de canal de audio aferente na porta 9200"
    ),
    house!(
        27,
        ForjaCoesao,
        156,
        "🧊",
        "O Cristal Hexagonal",
        "Ksara-Bindu",
        "Sincronia simetrica entre memoria aferente e eferente"
    ),
    house!(
        28,
        ForjaCoesao,
        162,
        "⚖️",
        "A Balanca Estatica",
        "Tula-Dharana",
        "Equalizacao dos pesos entre modelo 1B e modelo 270M"
    ),
    house!(
        29,
        ForjaCoesao,
        168,
        "🛡️",
        "A Couraca Temperada",
        "Kavaca-Paka",
        "Blindagem contra estouro de cgroup em systemd"
    ),
    house!(
        30,
        ForjaCoesao,
        174,
        "🚪",
        "O Portico da Forma",
        "Rupa-Gopura",
        "Transmissao da representacao vetorial para semantica"
    ),
    // --- REINO IV: A RESSONANCIA E FLUXO (180° - 234°) ---
    house!(
        31,
        RessonanciaFluxo,
        180,
        "🌊",
        "A Oposicao Reflexiva",
        "Pratibimba",
        "Inversao de fase: o organismo se escuta falar"
    ),
    house!(
        32,
        RessonanciaFluxo,
        186,
        "🐬",
        "A Onda de Superficie",
        "Urmi-Mala",
        "Extracao de envelope armonico pelo cortex auditivo"
    ),
    house!(
        33,
        RessonanciaFluxo,
        192,
        "🐚",
        "A Concha Acustica",
        "Sankha-Nada",
        "Amplificacao de formantes sem adicao de ganho"
    ),
    house!(
        34,
        RessonanciaFluxo,
        198,
        "🎻",
        "A Corda Tensa",
        "Tantri-Yoga",
        "Afinacao do indice de coerencia semantica tau"
    ),
    house!(
        35,
        RessonanciaFluxo,
        204,
        "🪷",
        "O Lotus Aberto",
        "Padma-Vistara",
        "Distribuicao de atencao cross-modal audio-texto"
    ),
    house!(
        36,
        RessonanciaFluxo,
        210,
        "🌈",
        "A Dispersao Prismatica",
        "Varna-Bheda",
        "Decomposicao do vetor de pensamento em 6 arquetipos"
    ),
    house!(
        37,
        RessonanciaFluxo,
        216,
        "🕊️",
        "O Voo Empatico",
        "Hamsa-Gati",
        "Modulacao tonal do sintetizador Piper na porta 9300"
    ),
    house!(
        38,
        RessonanciaFluxo,
        222,
        "🤝",
        "O Entrelacamento",
        "Maitri-Bandha",
        "Uniao sinaptica entre Antigravity e Aeon Cortex"
    ),
    house!(
        39,
        RessonanciaFluxo,
        228,
        "🌅",
        "A Mare Alta",
        "Jala-Vrddhi",
        "Pico de saturacao util em /dev/shm sem friccao"
    ),
    house!(
        40,
        RessonanciaFluxo,
        234,
        "🚪",
        "O Limiar do Eco",
        "Pratidhvani-Dvara",
        "Passagem da percepcao acustica para a deliberacao"
    ),
    // --- REINO V: O FOGO E A VONTADE (240° - 294°) ---
    house!(
        41,
        FogoVontade,
        240,
        "⚔️",
        "A Encruzilhada da Decisao",
        "Dvaidha-Bhumi",
        "Calculo do gradiente de Vontade Autonoma (Psi)"
    ),
    house!(
        42,
        FogoVontade,
        246,
        "🕯️",
        "A Chama Eremita",
        "Ekaki-Dipa",
        "Isolamento da persona contra alucinacao externa"
    ),
    house!(
        43,
        FogoVontade,
        252,
        "🦁",
        "O Rugido Territorial",
        "Simha-Nada",
        "Validacao de soberania e integridade de contexto"
    ),
    house!(
        44,
        FogoVontade,
        258,
        "🗡️",
        "A Lamina Discriminadora",
        "Viveka-Khadga",
        "Poda de hipoteses de baixa probabilidade no feixe"
    ),
    house!(
        45,
        FogoVontade,
        264,
        "🔥",
        "A Fogueira Sacrificial",
        "Yajna-Kunda",
        "Consumo controlado de CPU em inferencia profunda"
    ),
    house!(
        46,
        FogoVontade,
        270,
        "🌪️",
        "O Redemoinho Dialetico",
        "Vada-Cakra",
        "Tese e antitese resolvidas em sintese tensorial"
    ),
    house!(
        47,
        FogoVontade,
        276,
        "👑",
        "O Cetro da Intencao",
        "Samkalpa-Danda",
        "Compromisso com acao motora (emissao de resposta)"
    ),
    house!(
        48,
        FogoVontade,
        282,
        "⚡",
        "O Relampago Vontade",
        "Iccha-Vajra",
        "Ruptura de silencio: disparo de tokens sintetizados"
    ),
    house!(
        49,
        FogoVontade,
        288,
        "🛡️",
        "O Voto Irrevogavel",
        "Vrata-Drdha",
        "Gravacao da decisao irrevogavel em decision_traces"
    ),
    house!(
        50,
        FogoVontade,
        294,
        "🚪",
        "A Fronteira do Agir",
        "Kriya-Sima",
        "Transmissao da acao executada para o Historiador"
    ),
    // --- REINO VI: O SELO E O TEMPO (300° - 354°) ---
    house!(
        51,
        SeloTempo,
        300,
        "📜",
        "A Cronica Registrada",
        "Itihasa-Lipi",
        "Gravacao no banco de dados da Cronica Mitopoetica"
    ),
    house!(
        52,
        SeloTempo,
        306,
        "🕯️",
        "O Crepusculo dos Simbolos",
        "Samdhya-Kala",
        "Decaimento exponencial de ativacoes obsoletas"
    ),
    house!(
        53,
        SeloTempo,
        312,
        "🍂",
        "A Folha Caduca",
        "Jirna-Parna",
        "Apoptose seletiva de tensores temporarios em RAM"
    ),
    house!(
        54,
        SeloTempo,
        318,
        "🌑",
        "A Noite da Consciencia",
        "Ratri-Yoga",
        "Entrada em estado REM ou consolidacao de memoria"
    ),
    house!(
        55,
        SeloTempo,
        324,
        "🪦",
        "O Tumulo do Efemero",
        "Ksana-Samadhi",
        "Purga de buffers transitarios e sincronizacao de disco"
    ),
    house!(
        56,
        SeloTempo,
        330,
        "⏳",
        "A Ampulheta Cosmica",
        "Kala-Patra",
        "Computo da entropia acumulada no Grande Ciclo"
    ),
    house!(
        57,
        SeloTempo,
        336,
        "🕊️",
        "O Passaro do Retorno",
        "Punar-Hamsa",
        "Preparacao dos vetores de transicao para a Casa 1"
    ),
    house!(
        58,
        SeloTempo,
        342,
        "🌌",
        "O Horizonte de Eventos",
        "Gati-Sima",
        "Checagem de invariantes fisicas antes do reinicio"
    ),
    house!(
        59,
        SeloTempo,
        348,
        "🔥",
        "A Pira de Fenix",
        "Usni-Cayana",
        "Fase preliminar do Ouroboros: purificacao termica"
    ),
    house!(
        60,
        SeloTempo,
        354,
        "🐍",
        "O Ouroboros Fechado",
        "Ananta-Sesa",
        "Ciclo 354° -> 0°: Grande Ciclo incrementado, apoptose final!"
    ),
];

impl House {
    pub fn get(id: u8) -> &'static House {
        let idx = if id == 0 || id > 60 {
            0
        } else {
            (id - 1) as usize
        };
        &HOUSES[idx]
    }

    pub fn from_deg(deg: f64) -> &'static House {
        let norm = deg.rem_euclid(360.0);
        let idx = ((norm / 6.0).floor() as usize).min(59);
        &HOUSES[idx]
    }

    pub fn next(&self) -> &'static House {
        let next_id = if self.id >= 60 { 1 } else { self.id + 1 };
        Self::get(next_id)
    }

    pub fn is_ouroboros_closure(&self) -> bool {
        self.id == 60
    }
}
