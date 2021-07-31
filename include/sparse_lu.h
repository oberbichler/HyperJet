#pragma once

#include "hyperjet.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace hyperjet {

template <typename Scalar>
class SparseLU
{
private: // types

    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Sparse = Eigen::SparseMatrix<Scalar>;
    using Solver = Eigen::SparseLU<Sparse>;

private: // variables

    Vector m_rhs;
    Sparse m_lhs;
    Solver m_solver;
    Vector m_x;

public: // constructors

    SparseLU(index size) : m_rhs(size), m_lhs(size, size), m_x(size)
    {
        assert(!Scalar.is_dynamic());

        m_rhs.setZero();
        m_x.setZero();
    }

    SparseLU(index s, index size) : m_rhs(size), m_lhs(size, size), m_x(size)
    {
        assert(Scalar.is_dynamic());

        m_rhs.setZero();
        m_x.setZero();
    }

public: // methods

    Scalar lhs(index row, index col) const
    {
        return m_lhs.coeff(col, row);
    }

    Scalar rhs(index row) const
    {
        return m_rhs(row);
    }

    Scalar x(index row) const
    {
        return m_x(row);
    }

    void set_lhs(index row, index col, Scalar value)
    {
        m_lhs.coeffRef(col, row) = value;
    }

    void set_lhs(index row, index col, double value)
    {
        m_lhs.coeffRef(col, row) = value;
    }

    void set_rhs(index row, Scalar value)
    {
        m_rhs(row) = value;
    }

    void set_rhs(index row, double value)
    {
        m_rhs(row) = value;
    }

    void add_lhs(index row, index col, Scalar value)
    {
        m_lhs.coeffRef(col, row) += value;
    }

    void add_lhs(index row, index col, double value)
    {
        m_lhs.coeffRef(col, row) += value;
    }

    void add_rhs(index row, Scalar value)
    {
        m_rhs(row) += value;
    }

    void add_rhs(index row, double value)
    {
        m_rhs(row) += value;
    }

    void compress()
    {
        m_lhs.makeCompressed();
    }

    void clear()
    {
        for (int k = 0; k < m_lhs.outerSize(); ++k) {
            for (Sparse::InnerIterator it(m_lhs, k); it; ++it) {
                it.valueRef().set_zero();
            }
        }
        m_rhs.setZero();
        m_x.setZero();
    }

    void analyze_pattern()
    {
        m_solver.analyzePattern(m_lhs);
    }

    bool factorize()
    {
        m_solver.factorize(m_lhs);

        if (m_solver.info() != Eigen::Success) {
            return false;
        }

        return true;
    }

    bool solve()
    {
        m_x = m_solver.solve(m_rhs);

        if (m_solver.info() != Eigen::Success) {
            return false;
        }

        return true;
    }
};

} // namespace hyperjet