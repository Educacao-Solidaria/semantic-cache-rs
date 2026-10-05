//! Logging estruturado com `tracing`.
//!
//! Os logs vão para **stderr**: no transporte stdio do MCP, o stdout é o
//! canal do protocolo e qualquer linha de log ali corromperia as mensagens.
//!
//! No formato JSON cada evento carrega o span atual e a pilha de spans, e o
//! fechamento de cada span emite `time.busy`/`time.idle` — a latência de uma
//! operação assíncrona sai direto do log, mesmo atravessando `.await`s.

use tracing::Subscriber;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

use crate::config::{LogConfig, LogFormat};

#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    #[error("diretiva de log inválida: {0}")]
    Filter(#[from] tracing_subscriber::filter::ParseError),
    #[error("subscriber global já instalado: {0}")]
    Init(#[from] tracing::subscriber::SetGlobalDefaultError),
}

/// Monta o subscriber descrito em `config`, escrevendo em `writer`.
///
/// Separado de [`init`] para que testes capturem a saída sem instalar um
/// subscriber global.
///
/// # Errors
/// [`LoggingError::Filter`] se `config.level` não for uma diretiva válida.
pub fn subscriber<W>(
    config: &LogConfig,
    writer: W,
) -> Result<Box<dyn Subscriber + Send + Sync>, LoggingError>
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    let builder = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_new(&config.level)?)
        .with_writer(writer)
        .with_span_events(FmtSpan::CLOSE);
    Ok(match config.format {
        LogFormat::Json => Box::new(
            builder
                .json()
                .flatten_event(true)
                .with_current_span(true)
                .with_span_list(true)
                .finish(),
        ),
        LogFormat::Pretty => Box::new(builder.finish()),
    })
}

/// Instala o subscriber global, escrevendo em stderr. Chamar uma vez, no
/// início do `main`.
///
/// # Errors
/// Diretiva inválida ou subscriber global já instalado.
pub fn init(config: &LogConfig) -> Result<(), LoggingError> {
    tracing::subscriber::set_global_default(subscriber(config, std::io::stderr)?)?;
    Ok(())
}
