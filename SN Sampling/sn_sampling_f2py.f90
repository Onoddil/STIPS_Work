subroutine psf_fit_min(p, x, y, z, o_inv_sq, nitem, fun, jac, lenp, lenx, leny)
    implicit none
    integer, intent(in) :: nitem, lenp, lenx, leny
    double precision, intent(in) :: p(lenp), x(lenx), y(leny), z(lenx, leny), o_inv_sq(lenx, leny)
    double precision, intent(out) :: fun, jac(lenp)
    integer :: i, j, k, k_, nset
    double precision :: c, s, mux, muy, pi, f, common_terms
    double precision, allocatable :: g_minus_c(:)

    pi = 3.141592653589793
    ! p should be sets of (mux, muy, sigma, c) which should be referenced directly
    nset = lenp / nitem ! division of two integers in fortran is the same as a // b in python
    allocate(g_minus_c(nset))
    fun = 0.0d0
    jac(:) = 0.0d0
    do i = 1, leny
        do j = 1, lenx
            f = 0.0d0
            g_minus_c(:) = 0.0d0
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                mux = p(k_)
                muy = p(k_ + 1)
                s = p(k_ + 2)
                c = p(k_ + 3)
                g_minus_c(k) = 1.0d0 / (2.0d0 * pi * s) * exp(-0.5 * ((x(j) - mux)**2 + (y(i) - muy)**2) / s**2)
                f = f + c * g_minus_c(k)
            end do
            common_terms = 2 * (f - z(j, i)) * o_inv_sq(j, i)
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                mux = p(k_)
                muy = p(k_ + 1)
                s = p(k_ + 2)
                c = p(k_ + 3)
                jac(k_) = jac(k) + common_terms * f * (x(j) - mux) / s**2
                jac(k_ + 1) = jac(k_ + 1) + common_terms * f * (y(i) - muy) / s**2
                jac(k_ + 2) = jac(k_ + 2) + common_terms * f * ((x(i) - mux)**2 / s**2 + &
                                            (y(j) - muy)**2 / s**2 - 1) / s
                jac(k_ + 3) = jac(k_ + 3) + common_terms * g_minus_c(k)
            end do
            fun = fun + (f - z(j, i))**2 * o_inv_sq(j, i)
        end do
    end do

end subroutine psf_fit_min

subroutine psf_fit_hess(p, x, y, z, o_inv_sq, nitem, hess, lenp, lenx, leny)
    implicit none
    integer, intent(in) :: nitem, lenp, lenx, leny
    double precision, intent(in) :: p(lenp), x(lenx), y(leny), z(lenx, leny), o_inv_sq(lenx, leny)
    double precision, intent(out) :: hess(lenp, lenp)
    integer :: i, j, k, l, nset, k_, l_
    double precision :: c, s, mux, muy, pi, f, common_terms
    double precision, allocatable :: g_minus_c(:), dfda(:, :)

    pi = 3.141592653589793
    ! p should be sets of (mux, muy, sigma, c) which should be referenced directly
    nset = lenp / nitem ! division of two integers in fortran is the same as a // b in python
    hess(:, :) = 0.0d0
    allocate(g_minus_c(nset))
    allocate(dfda(nitem, nset))
    do i = 1, leny
        do j = 1, lenx
            dfda(:, :) = 0.0d0
            f = 0.0d0
            do k = 1, nset
                mux = p(k)
                muy = p(k + 1)
                s = p(k + 2)
                c = p(k + 3)
                g_minus_c(k) = 1.0d0 / (2.0d0 * pi * s) * exp(-0.5 * ((x(j) - mux)**2 + (y(i) - muy)**2) / s**2)
                f = f + c * g_minus_c(k)
            end do
            do k = 1, nset
                mux = p(k)
                muy = p(k + 1)
                s = p(k + 2)
                c = p(k + 3)
                common_terms = 2.0d0 * (f - z(j, i)) * o_inv_sq(j, i)
                dfda(1, k) = common_terms * f * (x(j) - mux) / s**2
                dfda(2, k) = common_terms * f * (y(i) - muy) / s**2
                dfda(3, k) = common_terms * f * ((x(i) - mux)**2 / s**2 + (y(j) - muy)**2 / s**2 - 1) / s
                dfda(4, k) = common_terms * g_minus_c(k)
            end do
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                do l = 1, nset
                    l_ = nitem*(l-1) + 1
                    common_terms = 2.0d0 * o_inv_sq(j, i)
                    hess(l_, k_) = hess(l_, k_) + common_terms * dfda(1, k) * dfda(1, l)
                    hess(l_, k_ + 1) = hess(l_, k_ + 1) + common_terms * dfda(2, k) * dfda(1, l)
                    hess(l_, k_ + 2) = hess(l_, k_ + 2) + common_terms * dfda(3, k) * dfda(1, l)
                    hess(l_, k_ + 3) = hess(l_, k_ + 3) + common_terms * dfda(4, k) * dfda(1, l)
                    hess(l_ + 1, k_) = hess(l_ + 1, k_) + common_terms * dfda(1, k) * dfda(2, l)
                    hess(l_ + 1, k_ + 1) = hess(l_ + 1, k_ + 1) + common_terms * dfda(2, k) * dfda(2, l)
                    hess(l_ + 1, k_ + 2) = hess(l_ + 1, k_ + 2) + common_terms * dfda(3, k) * dfda(2, l)
                    hess(l_ + 1, k_ + 3) = hess(l_ + 1, k_ + 3) + common_terms * dfda(4, k) * dfda(2, l)
                    hess(l_ + 2, k_) = hess(l_ + 2, k_) + common_terms * dfda(1, k) * dfda(3, l)
                    hess(l_ + 2, k_ + 1) = hess(l_ + 2, k_ + 1) + common_terms * dfda(2, k) * dfda(3, l)
                    hess(l_ + 2, k_ + 2) = hess(l_ + 2, k_ + 2) + common_terms * dfda(3, k) * dfda(3, l)
                    hess(l_ + 2, k_ + 3) = hess(l_ + 2, k_ + 3) + common_terms * dfda(4, k) * dfda(3, l)
                    hess(l_ + 3, k_) = hess(l_ + 3, k_) + common_terms * dfda(1, k) * dfda(4, l)
                    hess(l_ + 3, k_ + 1) = hess(l_ + 3, k_ + 1) + common_terms * dfda(2, k) * dfda(4, l)
                    hess(l_ + 3, k_ + 2) = hess(l_ + 3, k_ + 2) + common_terms * dfda(3, k) * dfda(4, l)
                    hess(l_ + 3, k_ + 3) = hess(l_ + 3, k_ + 3) + common_terms * dfda(4, k) * dfda(4, l)
                    if (k == l) then
                        mux = p(k_)
                        muy = p(k_ + 1)
                        s = p(k_ + 2)
                        c = p(k_ + 3)
                        common_terms = (f - z(j, i)) * o_inv_sq(j, i)
                        hess(l_, k_) = hess(l_, k_) + common_terms * &
                                                  f / s**2 * (((x(j) - mux) / s)**2 - 1)
                        hess(l_, k_ + 1) = hess(l_, k_ + 1) + common_terms * &
                                                          f * (x(j) - mux) * (y(i) - muy) / s**4
                        hess(l_, k_ + 2) = hess(l_, k_ + 2) + common_terms * &
                                         f * (x(j) - mux) * (((x(j) - mux) / s)**2 + ((y(i) - muy) / s)**2 - 1) / s**3
                        hess(l_, k_ + 3) = hess(l_, k_ + 3) + common_terms * &
                                                          g_minus_c(k) * (x(j) - mux) / s**2
                        hess(l_ + 1, k_) = hess(l_ + 1, k_) + common_terms * &
                                                          f * (x(j) - mux) * (y(i) - muy) / s**4
                        hess(l_ + 1, k_ + 1) = hess(l_ + 1, k_ + 1) + common_terms * &
                                                                  f / s**2 * (((y(i) - muy) / s)**2 - 1)
                        hess(l_ + 1, k_ + 2) = hess(l_ + 1, k_ + 2) + common_terms * &
                                         f * (y(i) - muy) * (((x(j) - mux) / s)**2 + ((y(i) - muy) / s)**2 - 1) / s**3
                        hess(l_ + 1, k_ + 3) = hess(l_ + 1, k_ + 3) + common_terms * &
                                                                  g_minus_c(k) * (y(i) - muy) / s**2
                        hess(l_ + 2, k_) = hess(l_ + 2, k_) + common_terms * &
                                         f * (x(j) - mux) * (((x(j) - mux) / s)**2 + ((y(i) - muy) / s)**2 - 1) / s**3
                        hess(l_ + 2, k_ + 1) = hess(l_ + 2, k_ + 1) + common_terms * &
                                         f * (y(i) - muy) * (((x(j) - mux) / s)**2 + ((y(i) - muy) / s)**2 - 1) / s**3
                        hess(l_ + 2, k_ + 2) = hess(l_ + 2, k_ + 2) + common_terms * &
                                                f / s**2 * (1 - (((x(i) - mux)/s)**2 + ((y(j) - muy)/s)**2 - 1)**2 - &
                                                3 * (((x(i) - mux)/s)**2 + ((y(j) - muy)/s)**2))
                        hess(l_ + 2, k_ + 3) = hess(l_ + 2, k_ + 3) + common_terms * &
                                                    g_minus_c(k) * (((x(i) - mux)/s)**2 + ((y(j) - muy)/s)**2 - 1) / s
                        hess(l_ + 3, k_) = hess(l_ + 3, k_) + common_terms * &
                                                          g_minus_c(k) * (x(j) - mux) / s**2
                        hess(l_ + 3, k_ + 1) = hess(l_ + 3, k_ + 1) + common_terms * &
                                                                  g_minus_c(k) * (y(i) - muy) / s**2
                        hess(l_ + 3, k_ + 2) = hess(l_ + 3, k_ + 2) + common_terms * &
                                                    g_minus_c(k) * (((x(i) - mux)/s)**2 + ((y(j) - muy)/s)**2 - 1) / s
                        ! no 3, 3 as d2fdc2 = 0
                    end if
                end do
            end do
        end do
    end do

end subroutine psf_fit_hess