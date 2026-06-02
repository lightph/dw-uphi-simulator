#pragma once

#include <complex>
#include <memory>
#include <vector>

#ifdef DOUBLE_PRECISION
using SIM_REAL = double;
using SIM_COMPLEX = std::complex<double>;
#else
using SIM_REAL = float;
using SIM_COMPLEX = std::complex<float>;
#endif

enum class ComputeBackend
{
    CPU,
    GPU
};

struct SimulatorConfig
{
    ComputeBackend backend;
    SIM_REAL alpha;
    SIM_REAL h;
    SIM_REAL h0;
    SIM_REAL Omega;
    SIM_REAL dt;
    SIM_REAL system_size;
    SIM_REAL min_res;
    SIM_REAL ic_amplitude;
    int ic_seed;
};

struct SpatialMetrics
{
    SIM_REAL mean_u;
    SIM_REAL mean_phi;
    SIM_REAL mean_phidot;
    SIM_REAL rugosity;
};

/**
 * @brief Interfaz abstracta pra las simulaciones.
 * * Define lo que se tiene que hacer,
 * pero no la implementacion en CPU o GPU
 */
class IDomainWallSimulator
{
public:
    virtual ~IDomainWallSimulator() = default;

    virtual void step() = 0;

    /**
     * @brief Ejecuta un bloque de pasos de la simulación
     * * En GPU se optimiza con graphs de CUDA
     * * @param steps Número de pasos a avanzar en bloque
     */
    virtual void run_block(size_t steps) = 0;
    virtual void create_step_graph(size_t steps) = 0;

    /**
     * @brief Calcula métricas espaciales (rugosidad, centro de masa) optimizado
     * para GPU o CPU
     * @return SpatialMetrics con mean_u, mean_phi y rugosity
     */
    virtual SpatialMetrics get_spatial_metrics() const = 0;

    // =========================================================================
    // HIGH-EFFICIENCY LOGGING & SPECTRAL ACCUMULATION
    // =========================================================================

    /**
           * @brief Habilita/deshabilita el registro fino de mean_phidot y rugosity durante run_block.
           * @param enable True para empezar a grabar, False para detener.
           * @param sample_every_n_steps Permite submuestrear (ej. grabar cada 10 pasos) para ahorrar memoria.
           */
    virtual void set_fine_tracking(bool enable, size_t sample_every_n_steps = 1) = 0;

    /**
             * @brief Obtiene el historial de mean_phidot registrado desde la última vez que se limpió.
             * Idealmente, el backend vacía su buffer interno después de esta llamada.
             */
    virtual std::vector<SIM_REAL>
    get_and_clear_fine_phi_dot_history() = 0;

    /**
             * @brief Obtiene el historial de rugosity registrado desde la última vez que se limpió.
             * Idealmente, el backend vacía su buffer interno después de esta llamada.
             */
    virtual std::vector<SIM_REAL>
    get_and_clear_fine_rugosity_history() = 0;

    /**
             * @brief Calcula el espectro de potencia de u (|FFT(u)|^2) en el device y
             * lo suma a un acumulador interno. No transfiere memoria al host.
             */
    virtual void
    accumulate_u_power_spectrum() = 0;

    /**
     * @brief Obtiene el promedio temporal del espectro de potencia acumulado.
     * Realiza una sola transferencia Device -> Host.
     */
    virtual std::vector<SIM_REAL> get_averaged_u_power_spectrum() const = 0;

    /**
     * @brief Reinicia a cero el acumulador interno del espectro de potencia.
     */
    virtual void reset_spectrum_accumulator() = 0;

    // =========================================================================

    virtual void get_z_host(std::vector<SIM_COMPLEX> &out) const = 0;
    virtual void get_zhat_host(std::vector<SIM_COMPLEX> &out) const = 0;

    virtual void set_h(SIM_REAL h) = 0;
    virtual void set_alpha(SIM_REAL alpha) = 0;
    virtual void set_h0(SIM_REAL h0) = 0;
    virtual void set_Omega(SIM_REAL Omega) = 0;
    virtual void set_dt(SIM_REAL dt) = 0;

    virtual SIM_REAL get_time() const = 0;
    virtual void set_time(SIM_REAL t) = 0;
    virtual SIM_REAL get_dt() const = 0;
    virtual SIM_REAL get_Omega() const = 0;

    virtual SIM_REAL get_h() const = 0;
    virtual SIM_REAL get_h0() const = 0;
    virtual SIM_REAL get_alpha() const = 0;
    virtual SIM_REAL get_system_size() const = 0;
    virtual size_t get_n_samples() const = 0;

    virtual void reset(SIM_REAL ic_amplitude, int ic_seed) = 0;
};

std::unique_ptr<IDomainWallSimulator>
create_simulator(const SimulatorConfig &config);